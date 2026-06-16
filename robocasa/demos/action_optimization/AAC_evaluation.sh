cd robocasa/demos/action_optimization

# -----------------------------------------------------1-8--------------------------------------------#
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task PnPCounterToCab --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task PnPCounterToStove --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081  --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task PnPCounterToSink --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task PnPCounterToMicrowave --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0

python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task PnPSinkToCounter --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task PnPStoveToCounter --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task PnPCabToCounter --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task PnPMicrowaveToCounter --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0

# -------------------------------------------------9-16--------------------------------------------------------#
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task CoffeeSetupMug --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task CoffeeServeMug --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081  --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task CloseDoubleDoor --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task OpenDoubleDoor --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0 --max_steps 1200

python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task OpenSingleDoor --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task CloseSingleDoor --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task OpenDrawer --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task CloseDrawer --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0

# -------------------------------------------------------17-24--------------------------------------------------------#
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task TurnSinkSpout --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task TurnOnSinkFaucet --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081  --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task TurnOffStove --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task CoffeePressButton --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0

python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task TurnOffSinkFaucet --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task TurnOnStove --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081  --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task TurnOnMicrowave --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0
python inference_client_action_entropy.py --out_path /home/sangfor/Videos/robocasa/action_entropy_test/ --task TurnOffMicrowave --chunk_size_selector gaussian_bernoulli --chunk_id_selector 0 --port 8081 --move_th 3.0