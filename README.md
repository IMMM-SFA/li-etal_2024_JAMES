_your zenodo badge here_

# li-etal_2024_james

**Structural uncertainty in the sensitivity of urban temperatures to anthropogenic heat flux**

Dan Li<sup>1\*</sup>, Ting Sun<sup>2</sup>, Jiachuan Yang <sup>3</sup>, Ning Zhang <sup>4</sup>, Pouya Vahmani <sup>5</sup>, and Andrew Jones <sup>5</sup>

<sup>1 </sup> Department of Earth and Environment and Department of Mechanical Engineering, Boston University, Boston, USA
<sup>2 </sup> Institute for Risk and Disaster Reduction, University College London, London, UK
<sup>3 </sup> Department of Civil and Environmental Engineering, The Hong Kong University of Science and Technology, Hong Kong, China
<sup>4 </sup> School of Atmospheric Sciences, Nanjing University, Nanjing, China
<sup>5 </sup> Lawrence Berkeley National Laboratory, Berkeley, USA

\* corresponding author:  lidan@bu.edu

## Abstract
One key source of uncertainty for weather and climate models is structural uncertainty, yet it is rarely examined in the context of simulated effects of anthropogenic heat flux in cities.
Using the Weather Research and Forecasting (WRF) model coupled with a single-layer urban canopy model (SLUCM), it is found that the sensitivity of urban canopy air temperature to anthropogenic heat flux can differ by an order of magnitude depending on how anthropogenic heat flux is handled.
Moreover, key model choices such as the treatment of roof-air interaction and the parameterization of convective heat transfer between the canopy air and the atmosphere can affect the sensitivity of urban canopy air temperature by a factor of 4.
Urban surface temperature and 2-m air temperature are less sensitive to the methods of handling anthropogenic heat flux and the examined model choices than urban canopy air temperature, but their sensitivities to anthropogenic heat flux can still vary by as much as a factor of 4 for surface temperature and 2 for 2-m air temperature.
Our study recommends using temperature sensitivity instead of temperature response to understand how various physical processes (and their representations in numerical models) modulate the simulated effects of anthropogenic heat flux.

## Journal reference
Li et al. (in submission). Structural uncertainty in the sensitivity of urban temperatures to anthropogenic heat flux

## Code reference

https://github.com/DanLi-BU/WRF/tree/WRF_AH (tag: AH_final)

## Data reference

### Input data

https://www2.mmm.ucar.edu/wrf/users/download/get_sources_wps_geog.html
https://www.ncei.noaa.gov/products/weather-climate-models/north-american-regional

### Output data



## Contributing modeling software
| Model | Version | Repository Link | Tag |
|-------|---------|-----------------|-----|
| WRF | 4.2.2 | https://github.com/DanLi-BU/WRF/tree/WRF_AH | AH_final |

## Reproduce my experiment
Fill in detailed info here or link to other documentation that is a thorough walkthrough of how to use what is in this repository to reproduce your experiment.


1. Install the software components required to conduct the experiement from [Contributing modeling software](#contributing-modeling-software)
2. Download and install the supporting input data required to conduct the experiement from [Input data](#input-data)
3. Run the following scripts in the `workflow` directory to re-create this experiment:

| Script Name | Description | How to Run |
| --- | --- | --- |
| `namelist.wps` | namelist to run the WPS part of my experiment |  |
| `namelist.input` | namelist to run the WRF part of my experiment |  |
| `URBPARM.TBL` | an example of URBPARM.TBL where the 3 key entires to change are AHOPTION, ROOF_TO_CANOPY_AIR_OPTION, and CH_SCHEME |  |

3.1 AHOPTION = 4 in URBPARM.TBL corresponds to method 1 in the manuscript, AHOPTION = 3 corresponds to method 2, AHOPTION = 2 corresponds to method 3, AHOPTION = 5 corresponds to revised method 1
3.2 ROOF_TO_CANOPY_AIR_OPTION = 0 in URBPARM.TBL corresponds to cases 1 and 3 in the manuscript, ROOF_TO_CANOPY_AIR_OPTION = 1 corresponds to cases 2 and 4
3.3 CH_SCHEME = 100 in URBPARM.TBL corresponds to 3 in the manuscript, CH_SCHEME = 2 corresponds to cases 1 and 2

## Reproduce my figures
Use the scripts found in the `figures` directory to reproduce the figures used in this publication. See details in the readme.docx document in the figures directory.
