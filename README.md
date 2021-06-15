# Are Psychiatric Disorders as Seasonal as Infections?

It is widely accepted that many infectious diseases are seasonal, though no thorough investigation has been done to probe the seasonality of other disease systems. In this study, we designed a statistical model to infer the seasonality of 33 neuropsychiatric and 47 infectious diseases in both the US and Sweden, based on two large-scale, nation-wide databases. The results indicate that neuropsychiatric diseases may be as seasonal as infectious diseases. In Sweden, the variation of neuropsychiatric diseases is much larger than in the US whereas the seasonal variations of infectious diseases in these two nations are similar, potentially pinning the pathogenesis of neuropsychiatric diseases on circadian rhythms. The study thus suggests that we should contemplate the influence of the sleep-wake cycle, light exposure, and circadian rhythms on the development of neuropsychiatric diseases.

## Data

The data we used for analyses can be found in the <tt>data</tt> directory, orangized by geographic regions (four high-latitude states, two low-latitude states, and the whole US). The data is packaged in a hierarchical python dictionary.

```python
import pickle

# cond is a python dict
# It has two parts: Condition and Data

with open('cond_1.bpkl3', 'rb') as f:
    cond = pickle.load(f)

# Females' Abnormal spine curvature, a disease of the musculoskeletal system
# Out: Abnormal_spine_curvature|musculoskeletal$female.bpkl3
print(cond['Condition'])

# cond_data is a python defaultdict, length = 600
# It contains the data of 600 population strata (patients who joined and left the register together) 
cond_data = cond['Data']

# Out: 600
print(len(cond_data))

# An example key of the 600 population stratum
# Out: (1, 46, 'F', '51-65')
# meaning a female population stratum aged 51-65 enrolled in our data from week 1 to week 46
print(sorted(cond_data.keys())[0])
```


<p align="center">
  <img src="./us_seasonality_psy.png" width="800">
</p>

<p align="center">
  <img src="./migraine_us_female.png" width="600">
</p>


