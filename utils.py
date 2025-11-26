import ee
import requests
import zipfile
import time
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, List, Tuple, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


class OptimizedChirpsDownloader:
    """
    Optimized CHIRPS downloader with dynamic stacks
    """
    
    # GEE limits
    MAX_SIZE_MB = 32  # Limit per request in GEE
    SAFETY_FACTOR = 0.85  # Use 85% of the limit for safety
    
    def __init__(self, output_dir: str = "./chirps_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Directory for final TIFFs
        self.tiff_dir = self.output_dir / "tiffs"
        self.tiff_dir.mkdir(exist_ok=True)
        
        # Temporary directory for ZIPs
        self.temp_dir = self.output_dir / "temp_zips"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Operation log
        self.download_log = []
        
    
    def get_country_geometry(self, country_name):
        """Gets geometry for a country"""
        countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
        country = countries.filter(ee.Filter.eq('ADM0_NAME', country_name))
        return country.geometry()
    
    def _build_points_fc(
        self,
        points: tuple[float, float] | tuple[tuple[float, float], ...]
    ) -> ee.FeatureCollection:
        """
        Creates a FeatureCollection of points with properties:
        - point_id
        - lon
        - lat
        """
        first = points[0]
        if isinstance(first, (int, float)):
            pts = [points]  # single point
        else:
            pts = list(points)  # multiple points

        features = []
        for idx, (lon, lat) in enumerate(pts):
            geom = ee.Geometry.Point([lon, lat])
            feat = ee.Feature(
                geom,
                {
                    'point_id': idx,
                    'lon': float(lon),
                    'lat': float(lat),
                }
            )
            features.append(feat)

        return ee.FeatureCollection(features)

    
    def estimate_image_size_mb(self, region: ee.Geometry, scale: int = 5000) -> float:
        """
        Estimates the size of a CHIRPS image in MB
        
        Formula: pixels × bytes_per_pixel × compression_factor
        """
        # Compute area in km²
        area_km2 = region.area().divide(1e6).getInfo()
        
        # Compute number of pixels (scale in meters)
        pixel_area_km2 = (scale / 1000) ** 2
        num_pixels = area_km2 / pixel_area_km2
        
        # CHIRPS is Float32 (4 bytes) but compresses well (~0.3 factor)
        bytes_per_pixel = 4
        compression_factor = 0.3
        
        estimated_mb = (num_pixels * bytes_per_pixel * compression_factor) / (1024 ** 2)
        
        return estimated_mb
    
    
    def calculate_optimal_stack_size(self, 
                                     single_image_mb: float,
                                     total_images: int) -> int:
        """
        Calculates how many images can go into a single stack
        
        Returns:
            Optimal number of images per stack
        """
        max_allowed_mb = self.MAX_SIZE_MB * self.SAFETY_FACTOR
        
        # Calculate images per stack
        images_per_stack = int(max_allowed_mb / single_image_mb)
        
        # Minimum 1, maximum 50 (to avoid very large stacks in memory)
        images_per_stack = max(1, min(images_per_stack, 50))
        
        return images_per_stack
    
    
    def create_stack_groups(self,
                           dates: List[str],
                           images_per_stack: int) -> List[List[str]]:
        """
        Groups dates into stacks
        
        Returns:
            List of date groups
        """
        stacks = []
        for i in range(0, len(dates), images_per_stack):
            stack = dates[i:i + images_per_stack]
            stacks.append(stack)
        
        return stacks
    
    
    def parse_date_window(self, window_mmdd: List[str]) -> Tuple[str, str]:
        """Converts MM-DD window to a usable format"""
        start_md, end_md = window_mmdd
        return start_md, end_md
    
    
    def generate_date_list(self, 
                          years: List[int],
                          window_mmdd: List[str],
                          temp_target: Literal['daily', 'monthly']) -> List[str]:
        """Generates a date list according to parameters"""
        dates = []
        start_md, end_md = self.parse_date_window(window_mmdd)
        
        for year in range(years[0], years[1] + 1):
            start_date = datetime.strptime(f"{year}-{start_md}", "%Y-%m-%d")
            end_date = datetime.strptime(f"{year}-{end_md}", "%Y-%m-%d")
            
            if temp_target == 'monthly':
                current = start_date.replace(day=1)
                while current <= end_date:
                    dates.append(current.strftime("%Y-%m"))
                    if current.month == 12:
                        current = current.replace(year=current.year + 1, month=1)
                    else:
                        current = current.replace(month=current.month + 1)
            else:  # daily
                current = start_date
                while current <= end_date:
                    dates.append(current.strftime("%Y-%m-%d"))
                    current += timedelta(days=1)
        
        return dates
    
    
    def create_image_stack(self,
                          dates: List[str],
                          temp_target: Literal['daily', 'monthly'],
                          temp_agg: Literal['mean', 'min', 'max', 'sum'],
                          region: ee.Geometry) -> ee.Image:
        """
        Creates a multiband stack image for multiple dates
        
        Returns:
            ee.Image with one band per date
        """
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
        
        bands = []
        band_names = []
        
        for date in dates:
            if temp_target == 'daily':
                # Filter specific day
                img = chirps.filterDate(
                    date,
                    (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                ).first()
                
                band_name = f"chirps_{date.replace('-', '')}"
            
            else:  # monthly
                # Aggregate whole month
                parts = date.split('-')
                year, month = int(parts[0]), int(parts[1])
                
                month_start = ee.Date.fromYMD(year, month, 1)
                month_end = month_start.advance(1, 'month')
                
                monthly_data = chirps.filterDate(month_start, month_end)
                
                if temp_agg == 'mean':
                    img = monthly_data.mean()
                elif temp_agg == 'sum':
                    img = monthly_data.sum()
                elif temp_agg == 'min':
                    img = monthly_data.min()
                elif temp_agg == 'max':
                    img = monthly_data.max()
                
                band_name = f"chirps_{date.replace('-', '')}"
            
            bands.append(img.rename(band_name))
            band_names.append(band_name)
        
        # Combine into a multiband stack
        if len(bands) == 1:
            stack = bands[0]
        else:
            stack = bands[0]
            for band in bands[1:]:
                stack = stack.addBands(band)
        
        return stack.clip(region)
    
    
    def download_stack(self,
                      stack: ee.Image,
                      stack_id: str,
                      region: ee.Geometry,
                      scale: int = 5000,
                      max_retries: int = 3) -> Tuple[bool, str, float, str]:
        """
        Downloads a stack (can be ZIP or direct TIFF)
        
        Returns:
            (success, message, elapsed_time, file_type)
        """
        start_time = time.time()
        temp_path = self.temp_dir / f"{stack_id}.tmp"
        
        for attempt in range(max_retries):
            try:
                # Generate download URL
                url = stack.getDownloadURL({
                    'region': region,
                    'scale': scale,
                    'format': 'ZIPPED_GEO_TIFF',
                    'crs': 'EPSG:4326'
                })
                
                # Download
                response = requests.get(url, stream=True, timeout=600)
                response.raise_for_status()
                
                # Save temporarily
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Detect file type
                with open(temp_path, 'rb') as f:
                    magic = f.read(4)
                
                # ZIP: starts with 'PK\x03\x04'
                # TIFF: starts with 'II*\x00' (little-endian) or 'MM\x00*' (big-endian)
                if magic[:2] == b'PK':
                    file_type = 'zip'
                    final_path = self.temp_dir / f"{stack_id}.zip"
                elif magic[:2] in (b'II', b'MM'):
                    file_type = 'tiff'
                    final_path = self.temp_dir / f"{stack_id}.tif"
                else:
                    raise ValueError(f"Unknown format: {magic[:4]}")
                
                # Rename with correct extension
                temp_path.rename(final_path)
                
                elapsed = time.time() - start_time
                return True, f"✓ Stack {stack_id}", elapsed, file_type
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    elapsed = time.time() - start_time
                    return False, f"✗ Stack {stack_id}: {str(e)[:50]}", elapsed, 'unknown'
        
        return False, f"✗ Stack {stack_id}: Max retries", time.time() - start_time, 'unknown'
    
    def extract_stack(self, file_path: Path) -> List[Path]:
       
        """
    Extracts TIFF(s) from the ZIP or returns the TIFF as is.
    Does NOT split by bands.

    If it is a ZIP:
        - Extracts .tif/.tiff
        - Renames them using the ZIP name:
            zip_name.zip -> zip_name.tif      (if there is 1 tiff)
            zip_name.zip -> zip_name_1.tif,
                            zip_name_2.tif... (if there are several tiffs)
    """
        extracted_files = []
 
        try:
            # ZIP case
            if file_path.suffix.lower() == '.zip':
                zip_stem = file_path.stem  # base name of the zip, without extension

                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    tif_files = [
                        f for f in zip_ref.namelist()
                        if f.lower().endswith(('.tif', '.tiff'))
                    ]

                    for idx, tif_file in enumerate(tif_files, start=1):
                        # Extract to tiff_dir
                        zip_ref.extract(tif_file, self.tiff_dir)
                        extracted = self.tiff_dir / tif_file

                        # If it came inside a subfolder, move it to tiff_dir
                        if extracted.parent != self.tiff_dir:
                            flat_path = self.tiff_dir / extracted.name
                            extracted.rename(flat_path)
                            extracted = flat_path

                        # New name based on ZIP
                        if len(tif_files) == 1:
                            new_name = f"{zip_stem}.tif"
                        else:
                            new_name = f"{zip_stem}_{idx}.tif"

                        final_path = self.tiff_dir / new_name

                        # If it already exists, overwrite (simple behavior)
                        if final_path.exists():
                            final_path.unlink()

                        extracted.rename(final_path)
                        extracted_files.append(final_path)

                # Delete original ZIP
                file_path.unlink()

            # Direct TIFF case
            elif file_path.suffix.lower() in ('.tif', '.tiff'):
                extracted_files.append(file_path)

        except Exception as e:
            print(f"  ⚠️ Error processing {file_path.name}: {e}")

        return extracted_files
    
    
    def download_points_timeseries(
                                    self,
                                    points: tuple[float, float] | tuple[tuple[float, float], ...],
                                    years: List[int],
                                    window_mmdd: List[str],
                                    temp_target: Literal['daily', 'monthly'] = 'daily',
                                    temp_agg: Literal['mean', 'min', 'max', 'sum'] = 'sum',
                                    scale: int = 5000,
                                    file_name: str | None = None,
                                ) -> Path:
        """
        Downloads CHIRPS time series for one or several points as a long table.

        Row = point × date (day or month).

        Columns:
            point_id, lon, lat, date, precip
        """

        # 1) Points FeatureCollection
        points_fc = self._build_points_fc(points)

        # 2) Date list
        dates = self.generate_date_list(years, window_mmdd, temp_target)

        if file_name is None:
            file_name = f"chirps_points_{temp_target}_{years[0]}_{years[1]}"

        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')

        # =======================
        # DAILY CASE (row = point × day)
        # =======================
        if temp_target == 'daily':
            start_date = dates[0]
            end_date_dt = datetime.strptime(dates[-1], "%Y-%m-%d") + timedelta(days=1)
            end_date = end_date_dt.strftime("%Y-%m-%d")

            daily_coll = chirps.filterDate(start_date, end_date)

            def sample_daily(img):
                img_precip = img.select(0).rename('precip')
                date_str = img.date().format('YYYY-MM-dd')
                samples = img_precip.sampleRegions(
                    collection=points_fc,
                    scale=scale,
                    geometries=True
                )
                return samples.map(lambda f: f.set('date', date_str))

            fc_per_image = daily_coll.map(sample_daily)
            all_samples = ee.FeatureCollection(fc_per_image).flatten()

        # =======================
        # MONTHLY CASE (row = point × month)
        # =======================
        else:  # 'monthly'
            monthly_images = []

            for date_str in dates:  # date_str = 'YYYY-MM'
                year, month = map(int, date_str.split('-'))
                month_start = ee.Date.fromYMD(year, month, 1)
                month_end = month_start.advance(1, 'month')

                monthly_data = chirps.filterDate(month_start, month_end)

                if temp_agg == 'mean':
                    img = monthly_data.mean()
                elif temp_agg == 'sum':
                    img = monthly_data.sum()
                elif temp_agg == 'min':
                    img = monthly_data.min()
                elif temp_agg == 'max':
                    img = monthly_data.max()
                else:
                    raise ValueError(f"Unknown temp_agg: {temp_agg}")

                img = img.select(0).rename('precip').set('date_str', date_str)
                monthly_images.append(img)

            monthly_coll = ee.ImageCollection(monthly_images)

            def sample_monthly(img):
                date_str = ee.String(img.get('date_str'))
                samples = img.sampleRegions(
                    collection=points_fc,
                    scale=scale,
                    geometries=True
                )
                return samples.map(lambda f: f.set('date', date_str))

            fc_per_image = monthly_coll.map(sample_monthly)
            all_samples = ee.FeatureCollection(fc_per_image).flatten()

        # =======================
        # EXPORT TABLE (CSV)
        # =======================
        selectors = ['point_id', 'lon', 'lat', 'date', 'precip']

        url = all_samples.getDownloadURL(
            filetype='CSV',
            selectors=selectors,
            filename=file_name
        )

        out_path = self.output_dir / f"{file_name}.csv"

        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()

        with open(out_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✅ Point time series ({temp_target}) saved to: {out_path}")
        return out_path
    
    def download_chirps_optimized(self,
                                  points: tuple[float, float] | tuple[tuple[float, float], ...] | None = None,
                                  bbox: tuple[float, float, float, float] | None = None,
                                  shape: str = None,  
                                  years: List[int] = None,
                                  window_mmdd: List[str] = None,
                                  temp_target: Literal['daily', 'monthly'] = 'daily',
                                  temp_agg: Literal['mean', 'min', 'max', 'sum'] = 'sum',
                                  scale: int = 5000,
                                  max_workers: int = 5):
        """
        Optimized download with dynamic stacks and parallelism
        
        Args:
            region: Geometry of the region of interest
            years: [start_year, end_year]
            window_mmdd: ['MM-DD', 'MM-DD'] annual window
            temp_target: 'daily' or 'monthly'
            temp_agg: 'mean', 'sum', 'min', 'max'
            scale: Spatial resolution in meters
            max_workers: Parallel downloads (recommended: 2–4)
        """
        # 1) POINT MODE → long table
        if points is not None:
            print("\n" + "="*70)
            print("📍 POINT MODE: downloading point time series as long table")
            print("="*70)

            out_table = self.download_points_timeseries(
                points=points,
                years=years,
                window_mmdd=window_mmdd,
                temp_target=temp_target,
                temp_agg=temp_agg,
                scale=scale,
                file_name=None
            )
            return out_table  # Exit here, no stacks for points

        flags = [points is not None, bbox is not None, shape is not None]
        if sum(flags) == 0:
            raise ValueError("You must specify one of: points, bbox or shape.")
        if sum(flags) > 1:
            raise ValueError("You cannot combine points, bbox and shape at the same time.")

        # --- Build region (ee.Geometry) ---
        if bbox is not None:
            # Rectangle: (min_lon, min_lat, max_lon, max_lat)
            region = ee.Geometry.Rectangle(bbox)

        elif shape is not None:
            # Here we assume 'shape' is a country name; reuse helper
            region = self.get_country_geometry(shape)

        
        
        print("\n" + "="*70)
        print("🚀 OPTIMIZED CHIRPS DOWNLOAD WITH STACKS")
        print("="*70)
        
        # 1. Generate date list
        dates = self.generate_date_list(years, window_mmdd, temp_target)
        total_images = len(dates)
        print(f"\n📅 Total images: {total_images}")
        print(f"   Range: {dates[0]} → {dates[-1]}")
        
        # 2. Estimate size per image
        print(f"\n📊 Estimating download capacity...")
        single_image_mb = self.estimate_image_size_mb(region, scale)
        print(f"   Estimated size per image: {single_image_mb:.2f} MB")
        
        # 3. Compute optimal stacks
        images_per_stack = self.calculate_optimal_stack_size(single_image_mb, total_images)
        stacks = self.create_stack_groups(dates, images_per_stack)
        
        print(f"\n📦 Stack configuration:")
        print(f"   Images per stack: {images_per_stack}")
        print(f"   Estimated size per stack: {single_image_mb * images_per_stack:.2f} MB")
        print(f"   Total stacks: {len(stacks)}")
        print(f"   Parallel workers: {max_workers}")
        
        # Temporal summary
        if images_per_stack == 1:
            periodo = "1 image"
        elif temp_target == 'daily':
            dias = images_per_stack
            if dias >= 365:
                periodo = f"~{dias//365} years"
            elif dias >= 30:
                periodo = f"~{dias//30} months"
            else:
                periodo = f"{dias} days"
        else:  # monthly
            meses = images_per_stack
            if meses >= 12:
                periodo = f"~{meses//12} years"
            else:
                periodo = f"{meses} months"
        
        print(f"   Period per stack: {periodo}")
        
        # Overall size
        total_size_gb = (single_image_mb * total_images) / 1024
        print(f"\n💾 Total estimated size: {total_size_gb:.2f} GB")
        
        input("\n⏸️  Press ENTER to start download...")
        
        # 4. Prepare download tasks
        download_tasks = []
        for i, stack_dates in enumerate(stacks):
            stack_id = (
                f"stack_{i+1:04d}_"
                f"{stack_dates[0].replace('-', '')}_to_"
                f"{stack_dates[-1].replace('-', '')}"
            )
            
            download_tasks.append({
                'stack_id': stack_id,
                'dates': stack_dates,
                'index': i + 1,
                'total': len(stacks)
            })
        
        # 5. Parallel download
        print(f"\n{'='*70}")
        print("⬇️  STARTING DOWNLOAD")
        print("="*70)
        
        start_time = time.time()
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create stacks and download
            futures = {}
            
            for task in download_tasks:
                # Create stack
                stack = self.create_image_stack(
                    task['dates'],
                    temp_target,
                    temp_agg,
                    region
                )
                
                # Submit download
                future = executor.submit(
                    self.download_stack,
                    stack,
                    task['stack_id'],
                    region,
                    scale
                )
                futures[future] = task
            
            # Process results
            for future in as_completed(futures):
                task = futures[future]
                success, message, elapsed, file_type = future.result()
                
                # Progress display
                if success:
                    successful += 1
                    print(f"{message} ({task['index']}/{task['total']}) - {elapsed:.1f}s")
                else:
                    failed += 1
                    print(f"{message} ({task['index']}/{task['total']})")
                
                # Log
                self.download_log.append({
                    'stack_id': task['stack_id'],
                    'success': success,
                    'elapsed': elapsed,
                    'num_images': len(task['dates'])
                })
        
        total_download_time = time.time() - start_time
        
        # 6. Extract all ZIPs
        print(f"\n{'='*70}")
        print("📂 EXTRACTING FILES")
        print("="*70)
        
        zip_files = list(self.temp_dir.glob("*.zip"))
        total_extracted = 0
        
        for zip_file in zip_files:
            extracted = self.extract_stack(zip_file)
            total_extracted += len(extracted)
            print(f"  ✓ {zip_file.name} → {len(extracted)} TIFFs")
        
        # Clean up temp directory
        try:
            if self.temp_dir.exists() and not list(self.temp_dir.glob("*")):
                self.temp_dir.rmdir()
        except:
            pass
        
        # 7. Final summary
        print(f"\n{'='*70}")
        print("✅ DOWNLOAD COMPLETED")
        print("="*70)
        print(f"Successful stacks:  {successful}/{len(stacks)}")
        print(f"Failed stacks:      {failed}/{len(stacks)}")
        print(f"Extracted TIFFs:    {total_extracted}")
        print(f"Total time:         {total_download_time/60:.1f} minutes")
        print(f"Speed:              {total_images/total_download_time:.1f} imgs/min")
        print(f"\n📁 Files in: {self.tiff_dir}")
        print("="*70)
        
        # Save log
        log_path = self.output_dir / "download_log.json"
        with open(log_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_images': total_images,
                    'total_stacks': len(stacks),
                    'images_per_stack': images_per_stack,
                    'successful_stacks': successful,
                    'failed_stacks': failed,
                    'total_time_min': total_download_time / 60,
                    'images_per_min': total_images / total_download_time
                },
                'stacks': self.download_log
            }, f, indent=2)
        
        print(f"📊 Log saved to: {log_path}\n")

