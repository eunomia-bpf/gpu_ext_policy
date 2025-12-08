sudo python  /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/docs/eval/run_policy_comparison.py


sudo python run_policy_comparison.py --kernel hotspot --size-factor 0.6 --output results_hotspot
sudo python run_policy_comparison.py --kernel gemm --size-factor 0.6 --output results_gemm
sudo python run_policy_comparison.py --kernel kmeans_sparse --size-factor 0.9 --output results_kmeans
