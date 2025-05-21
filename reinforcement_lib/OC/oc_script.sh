export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=true

#args=("$@")
echo "Port Number is $1"
xvfb-run -a webots --mode=fast --stdout --stderr --batch --minimize --no-rendering --port=$1 StartingWorld.wbt &
#xvfb-run -a webots --mode=fast --stdout --stderr --batch --minimize --log-performance=/home/b/brendon45/oc_tests/output_logs/webots_perf.log --no-rendering StartingWorld.wbt &
export WEBOTS_HOME=/usr/local/webots
#$WEBOTS_HOME/webots-controller --port=$1 /home/b/brendon45/oc_tests/main.py ${@:2}
$WEBOTS_HOME/webots-controller --port=$1 /home/b/brendon45/oc_tests/experiment_oc.py #${@:2}
#$WEBOTS_HOME/webots-controller --port=$1 /home/b/brendon45/oc_tests/experiment_two_maze.py #${@:2}
#$WEBOTS_HOME/webots-controller --port=$1 /home/b/brendon45/oc_tests/experiment_no_term_ocv2.py #${@:2}
#$WEBOTS_HOME/webots-controller --port=$1 /home/b/brendon45/oc_tests/experiment_custom_oc.py ${@:2}
#$WEBOTS_HOME/webots-controller --port=$1 /home/b/brendon45/oc_tests/experiment_oc_no_pc.py ${@:2}
#$WEBOTS_HOME/webots-controller --port=${args[0]} /home/b/brendon45/oc_tests/main.py 
