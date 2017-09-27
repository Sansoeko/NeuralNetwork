#!/usr/bin/env bash
declare -a num_hidden_layers=(1 2 3 4 5 6 7 8 9)
declare -a num_hidden_neurons=(10 20 30 40 50)
for hidden_layers in "${num_hidden_layers[@]}"
do
    for hidden_neurons in "${num_hidden_neurons[@]}"
    do
        echo "$hidden_layers"
        echo "$hidden_neurons"
        nohup python mnist.py --num_hidden_layers "$hidden_layers" --num_hidden_neurons "$hidden_neurons" &
    done
    wait
done