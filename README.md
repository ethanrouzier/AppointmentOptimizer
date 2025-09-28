# Medical Appointment Optimization System

A Flask web application designed to optimize medical appointments to facilitate carpooling and reduce CO2 emissions in healthcare settings.

## Overview

This system analyzes medical appointment data to identify potential carpooling opportunities between patients. By optimizing appointment schedules within the same medical specialty, the application maximizes the number of viable carpooling pairs while minimizing travel distances and CO2 emissions.

## Features

- **Appointment Analysis**: Processes medical appointment data with patient locations and schedules
- **Geographic Optimization**: Uses postal codes to calculate distances between patient locations
- **Schedule Optimization**: Reorders appointments within the same medical specialty to maximize carpooling opportunities
- **CO2 Impact Calculation**: Estimates kilometers saved and CO2 emissions reduced through carpooling
- **3D Visualization**: Interactive 3D plots showing appointment distribution before and after optimization
- **Constraint Management**: Respects time and distance constraints for viable carpooling pairs

## Technical Specifications

### Requirements
- Python 3.7+
- Flask
- Pandas
- NumPy
- Matplotlib
- pgeocode

### Key Parameters
- Maximum carpooling distance: 10 km
- Maximum time difference: 60 minutes
- CO2 emission factor: 0.2 kg CO2/km
- Supported departments: 50 (Manche), 61 (Orne)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install flask pandas numpy matplotlib pgeocode
   ```
3. Place your appointment data in CSV format in the `data/` directory
4. Run the application:
   ```bash
   python app.py
   ```

## Data Format

The application expects a CSV file with the following columns:
- `DATE_RDV`: Appointment date and time
- `RDV`: Medical specialty/appointment type
- `NIP`: Patient identifier
- `CODEPOSTALCODECOMMUNE`: Postal code
- `VILLE`: City name

## Usage

1. Access the web interface at `http://127.0.0.1:4000`
2. Select a date for optimization (2022 data)
3. Configure the one-way distance to the medical site
4. Click "Optimize Appointments" to run the analysis
5. Review the results including:
   - Number of carpooling pairs identified
   - Kilometers and CO2 emissions saved
   - 3D visualizations of appointment distributions
   - Detailed carpooling pair information

## Algorithm

The optimization algorithm works in several stages:

1. **Data Processing**: Filters appointments by department and date
2. **Geocoding**: Converts postal codes to latitude/longitude coordinates
3. **Clustering**: Groups nearby appointments using distance-based clustering
4. **Schedule Optimization**: Reorders appointments within medical specialties to maximize carpooling opportunities
5. **Pair Matching**: Identifies viable carpooling pairs based on distance and time constraints
6. **Impact Calculation**: Estimates environmental benefits

## Output

The system provides:
- Before/after comparison metrics
- 3D visualizations of appointment distributions
- Detailed carpooling pair listings
- Environmental impact estimates (km saved, CO2 reduced)

## Limitations

- Currently optimized for French postal codes
- Limited to departments 50 and 61
- Requires appointments within the same medical specialty for optimization
- Patients with multiple appointments on the same day are locked to their original schedule
