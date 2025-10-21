### For data load
mkdir -p ./data && wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O ./data/cifar-10-python.tar.gz && tar -xvzf ./data/cifar-10-python.tar.gz -C ./data && rm ./data/cifar-10-python.tar.gz

source fed-learning-env/bin/activate

#To load the right compiler
module load gcc/10.2.0

