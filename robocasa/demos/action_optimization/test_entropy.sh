cd /home/sangfor/code/lyc/robocasa/robocasa/demos/action_optimization
python inference_client_action_entropy_multisample_baselines.py --task OpenDoubleDoor --port 8085 
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/gr00t15_action_entropy_test --task OpenDoubleDoor --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8085

python inference_client_action_entropy_multisample_baselines.py --task OpenDrawer --port 8086 
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/gr00t15_action_entropy_test --task OpenDrawer --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8086




