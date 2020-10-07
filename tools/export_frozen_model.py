from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

flags.DEFINE_string('savedmodel', 'model/yolov3/1/', 'path to saved model file')
flags.DEFINE_string('output', 'model/frozen_model/', 'path to frozon model')

def main(_argv):

    model = tf.saved_model.load(FLAGS.savedmodel)#tags='serve'
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
#     print(infer.structured_outputs)

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: infer(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(infer.inputs[0].shape, infer.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=FLAGS.output,
                      name="frozen_graph.pb",
                      as_text=False)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
