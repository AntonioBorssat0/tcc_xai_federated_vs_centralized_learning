# Resumo LIME - XGBoost Federado (Cyclic)
Instâncias explicadas (sample_count): 2000
Top 19 features (global, mean |weight|):
- Humidity3pm: 0.162130
- Pressure3pm: 0.086589
- WindGustSpeed: 0.078693
- Pressure9am: 0.064120
- Temp9am: 0.053181
- MaxTemp: 0.036177
- MinTemp: 0.035199
- Temp3pm: 0.033513
- Humidity9am: 0.025733
- Rainfall: 0.019604
- WindSpeed9am: 0.017726
- WindDir9am_cos: 0.016656
- WindDir9am_sin: 0.015707
- WindSpeed3pm: 0.014667
- WindGustDir_cos: 0.013497
- WindDir3pm_sin: 0.013412
- WindGustDir_sin: 0.012249
- WindDir3pm_cos: 0.011129
- RainToday: 0.007295

Diferenças mean |weight| (classe 0 - classe 1) (top 10 por abs diff):
- Humidity3pm: classe0=0.161566, classe1=0.164115, diff=-0.002550
- MaxTemp: classe0=0.035732, classe1=0.037743, diff=-0.002011
- Pressure3pm: classe0=0.087024, classe1=0.085056, diff=0.001968
- Temp3pm: classe0=0.033118, classe1=0.034901, diff=-0.001783
- WindDir9am_sin: classe0=0.015376, classe1=0.016870, diff=-0.001494
- WindGustSpeed: classe0=0.078395, classe1=0.079741, diff=-0.001345
- Humidity9am: classe0=0.025451, classe1=0.026727, diff=-0.001277
- WindDir3pm_sin: classe0=0.013168, classe1=0.014266, diff=-0.001098
- WindSpeed9am: classe0=0.017547, classe1=0.018355, diff=-0.000808
- Temp9am: classe0=0.053354, classe1=0.052575, diff=0.000779

Arquivos gerados:
- comparison_class_0_vs_1.csv
- feature_importance_all.csv
- feature_importance_class_0.csv
- feature_importance_class_1.csv
- lime_instance_abs_weights.csv
- lime_instance_fidelity.csv
- lime_instance_weights.csv
- lime_xgb_cyclic_class_0_mean_abs_weight_top19.png
- lime_xgb_cyclic_class_1_mean_abs_weight_top19.png
- lime_xgb_cyclic_feature_weight_distribution_top19.png
- lime_xgb_cyclic_global_mean_abs_weight_top19.png
- lime_xgb_cyclic_heatmap_top19_rows200.png
- lime_xgb_cyclic_heatmap_top19_rows300.png
- lime_xgb_cyclic_waterfall_idx109622_label1.png
- lime_xgb_cyclic_waterfall_idx14482_label0.png
- lime_xgb_cyclic_waterfall_idx61381_label0.png
- lime_xgb_cyclic_waterfall_idx68600_label1.png
- lime_xgb_cyclic_waterfall_idx94448_label0.png
- lime_xgb_cyclic_waterfall_idx97989_label1.png
- summary.md
