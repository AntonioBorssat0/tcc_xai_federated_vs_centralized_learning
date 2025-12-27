# Resumo LIME - Centralizado (MLP)
Instâncias explicadas (sample_count): 2000
Top 19 features (global, mean |weight|):
- Pressure3pm: 0.196463
- Pressure9am: 0.123458
- Humidity3pm: 0.120170
- WindGustSpeed: 0.091533
- Rainfall: 0.064448
- Temp3pm: 0.058444
- MinTemp: 0.053617
- Humidity9am: 0.041288
- MaxTemp: 0.040761
- WindSpeed3pm: 0.040714
- Temp9am: 0.024341
- WindDir3pm_sin: 0.020874
- WindDir9am_cos: 0.013770
- WindDir3pm_cos: 0.012074
- RainToday: 0.007969
- WindGustDir_sin: 0.007791
- WindSpeed9am: 0.007642
- WindDir9am_sin: 0.007525
- WindGustDir_cos: 0.007188

Diferenças mean |weight| (classe 0 - classe 1) (top 10 por abs diff):
- Temp9am: classe0=0.023492, classe1=0.027322, diff=-0.003830
- MaxTemp: classe0=0.040207, classe1=0.042709, diff=-0.002502
- Temp3pm: classe0=0.058097, classe1=0.059664, diff=-0.001567
- Pressure9am: classe0=0.123121, classe1=0.124640, diff=-0.001519
- Pressure3pm: classe0=0.196727, classe1=0.195535, diff=0.001192
- Humidity3pm: classe0=0.119910, classe1=0.121087, diff=-0.001177
- WindDir3pm_cos: classe0=0.011907, classe1=0.012660, diff=-0.000753
- MinTemp: classe0=0.053482, classe1=0.054091, diff=-0.000609
- WindDir9am_sin: classe0=0.007641, classe1=0.007118, diff=0.000523
- WindDir9am_cos: classe0=0.013686, classe1=0.014066, diff=-0.000381

Arquivos gerados:
- comparison_class_0_vs_1.csv
- feature_importance_all.csv
- feature_importance_class_0.csv
- feature_importance_class_1.csv
- lime_class_0_mean_abs_weight_top19.png
- lime_class_1_mean_abs_weight_top19.png
- lime_feature_weight_distribution_top19.png
- lime_global_mean_abs_weight_top19.png
- lime_heatmap_top19_rows200.png
- lime_heatmap_top19_rows300.png
- lime_instance_abs_weights.csv
- lime_instance_fidelity.csv
- lime_instance_weights.csv
- lime_waterfall_idx102488_label0.png
- lime_waterfall_idx14938_label1.png
- lime_waterfall_idx16801_label1.png
- lime_waterfall_idx61686_label0.png
- lime_waterfall_idx61994_label1.png
- lime_waterfall_idx72774_label0.png
- lime_waterfall_idx80585_label0.png
- lime_waterfall_idx99150_label1.png
- summary.md
