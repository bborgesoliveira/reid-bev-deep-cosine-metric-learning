# import json

# f = open('object_data.json')
# instances = json.load(f)

# total = 0
# object_dict = {}
# for token in instances:
#     if len(instances[token]['images']) >= 5:
#         total += 1
#         object_dict[token] = instances[token]
        
# with open('object_data_resumed.json', 'w') as f:
#     json.dump(object_dict, f, indent=4)
        

        
# print(f'Total: {total}')

# f.close()
import tensorflow as tf
print("Versão do TensorFlow:", tf.__version__)
print("CUDA disponível no TensorFlow:", tf.test.is_built_with_cuda())

# Exibe as versões do CUDA e do CuDNN que o TensorFlow está usando
for device in tf.config.list_physical_devices('GPU'):
    print(device)