#"""
#CNN Model for Remote Sensing Image Classification
#Traditional Convolutional Neural Network implementation
#"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
import time

class CNNModel:
    """Traditional CNN model for remote sensing classification"""
    
    def __init__(self, input_shape=(256, 256, 3), num_classes=5, architecture='custom'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        self.model = None
        self.training_history = None
        
    def build_custom_cnn(self, num_layers=6, base_filters=64, dropout_rate=0.3):
        """Build custom CNN architecture"""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape))
        
        # Convolutional blocks
        filters = base_filters
        for i in range(num_layers):
            # Convolutional layer
            model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
            model.add(layers.BatchNormalization())
            
            # Add max pooling every 2 layers
            if i % 2 == 1:
                model.add(layers.MaxPooling2D((2, 2)))
                model.add(layers.Dropout(dropout_rate))
                filters = min(filters * 2, 512)  # Cap at 512 filters
        
        # Global average pooling
        model.add(layers.GlobalAveragePooling2D())
        
        # Dense layers
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def build_pretrained_model(self, architecture='resnet50', freeze_layers=0):
        """Build model using pretrained backbone"""
        
        # Base model selection
        if architecture.lower() == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif architecture.lower() == 'efficientnet':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif architecture.lower() == 'densenet':
            base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Freeze layers if specified
        if freeze_layers > 0:
            for layer in base_model.layers[:freeze_layers]:
                layer.trainable = False
        
        # Add custom head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_model(self, **kwargs):
        """Build the model based on architecture choice"""
        if self.architecture == 'custom':
            self.model = self.build_custom_cnn(**kwargs)
        else:
            self.model = self.build_pretrained_model(self.architecture, **kwargs)
        
        return self.model
    
    def compile_model(self, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy'):
        """Compile the model"""
        if not self.model:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Configure optimizer
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def train(self, train_data, validation_data=None, epochs=50, batch_size=32, 
              callbacks_list=None, verbose=1):
        """Train the model"""
        if not self.model:
            raise ValueError("Model not built and compiled yet.")
        
        # Default callbacks
        if callbacks_list is None:
            callbacks_list = [
                callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
                callbacks.ModelCheckpoint('best_cnn_model.h5', save_best_only=True)
            ]
        
        # Training
        start_time = time.time()
        
        self.training_history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return self.training_history
    
    def predict(self, X, batch_size=32):
        """Make predictions"""
        if not self.model:
            raise ValueError("Model not trained yet.")
        
        start_time = time.time()
        predictions = self.model.predict(X, batch_size=batch_size)
        inference_time = time.time() - start_time
        
        return predictions, inference_time
    
    def evaluate(self, test_data, batch_size=32):
        """Evaluate model performance"""
        if not self.model:
            raise ValueError("Model not trained yet.")
        
        start_time = time.time()
        results = self.model.evaluate(test_data, batch_size=batch_size)
        evaluation_time = time.time() - start_time
        
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        metrics['evaluation_time'] = evaluation_time
        return metrics
    
    def get_feature_maps(self, X, layer_names=None):
        """Extract feature maps from intermediate layers"""
        if not self.model:
            raise ValueError("Model not built yet.")
        
        if layer_names is None:
            # Get some intermediate layers
            layer_names = [layer.name for layer in self.model.layers 
                          if 'conv' in layer.name.lower()][:3]
        
        # Create feature extraction model
        feature_extractor = models.Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer(name).output for name in layer_names]
        )
        
        features = feature_extractor(X)
        return features, layer_names
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if not self.model:
            raise ValueError("Model not built yet.")
        
        return self.model.summary()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.model:
            raise ValueError("Model not built yet.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.model = models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_energy_consumption(self, num_inferences=1000):
        """Estimate energy consumption (simplified)"""
        # Simplified energy estimation based on model parameters
        total_params = self.model.count_params()
        
        # Rough estimation: energy per parameter per inference (in microjoules)
        energy_per_param = 0.1  # μJ
        
        total_energy = total_params * energy_per_param * num_inferences
        return total_energy / 1000  # Convert to millijoules
    
    def benchmark_inference_speed(self, input_shape, num_runs=100):
        """Benchmark inference speed"""
        if not self.model:
            raise ValueError("Model not built yet.")
        
        # Create dummy input
        dummy_input = np.random.random((1,) + input_shape)
        
        # Warm up
        for _ in range(10):
            _ = self.model.predict(dummy_input, verbose=0)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.model.predict(dummy_input, verbose=0)
            times.append(time.time() - start_time)
        
        avg_time = np.mean(times) * 1000  # Convert to milliseconds
        std_time = np.std(times) * 1000
        
        return {
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'min_inference_time_ms': min(times) * 1000,
            'max_inference_time_ms': max(times) * 1000
        }


# Utility functions for remote sensing specific operations
def preprocess_remote_sensing_image(image, bands=['R', 'G', 'B'], normalize=True):
    """Preprocess remote sensing image"""
    
    # Band selection
    if len(bands) == 3 and image.shape[-1] >= 3:
        # RGB bands
        processed = image[:, :, :3]
    elif 'NIR' in bands and image.shape[-1] >= 4:
        # Include NIR band
        band_indices = []
        for band in bands:
            if band == 'R':
                band_indices.append(0)
            elif band == 'G':
                band_indices.append(1)
            elif band == 'B':
                band_indices.append(2)
            elif band == 'NIR':
                band_indices.append(3)
        processed = image[:, :, band_indices]
    else:
        processed = image
    
    # Normalization
    if normalize:
        processed = processed.astype(np.float32) / 255.0
    
    return processed

def calculate_spectral_indices(image):
    """Calculate common spectral indices"""
    # Assuming image has R, G, B, NIR bands
    if image.shape[-1] < 4:
        return None
    
    R = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    B = image[:, :, 2].astype(np.float32)
    NIR = image[:, :, 3].astype(np.float32)
    
    # NDVI (Normalized Difference Vegetation Index)
    ndvi = (NIR - R) / (NIR + R + 1e-8)
    
    # NDWI (Normalized Difference Water Index)
    ndwi = (G - NIR) / (G + NIR + 1e-8)
    
    # EVI (Enhanced Vegetation Index)
    evi = 2.5 * ((NIR - R) / (NIR + 6 * R - 7.5 * B + 1))
    
    return {
        'ndvi': ndvi,
        'ndwi': ndwi,
        'evi': evi
    }

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("CNN Model for Remote Sensing - Example Usage")
    
    # Create model
    cnn = CNNModel(input_shape=(256, 256, 4), num_classes=6, architecture='custom')
    
    # Build and compile
    model = cnn.build_model(num_layers=8, base_filters=64)
    cnn.compile_model(optimizer='adam', learning_rate=0.001)
    
    # Print model summary
    print("\nModel Architecture:")
    cnn.get_model_summary()
    
    # Benchmark inference speed
    print("\nBenchmarking inference speed...")
    speed_results = cnn.benchmark_inference_speed((256, 256, 4))
    print(f"Average inference time: {speed_results['avg_inference_time_ms']:.2f} ms")
    
    # Estimate energy consumption
    energy = cnn.get_energy_consumption(1000)
    print(f"Estimated energy consumption for 1000 inferences: {energy:.2f} mJ")
    
    print("\nCNN Model ready for remote sensing applications!")