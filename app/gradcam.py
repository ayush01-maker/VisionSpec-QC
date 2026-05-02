import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    """
    Generate Grad-CAM heatmap
    """

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap.numpy()


def overlay_gradcam(img_path, heatmap, alpha=0.4):
    """
    Overlay heatmap on original image
    """

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed_img = cv2.addWeighted(
        img,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return superimposed_img


def save_gradcam(model, img_array, img_path, output_path):
    """
    Generate and save Grad-CAM result
    """

    last_conv_layer_name = None

    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        raise ValueError("No convolutional layer found.")

    heatmap = get_gradcam_heatmap(
        model,
        img_array,
        last_conv_layer_name
    )

    result = overlay_gradcam(
        img_path,
        heatmap
    )

    cv2.imwrite(output_path, result)

    print(f"Grad-CAM saved at: {output_path}")
