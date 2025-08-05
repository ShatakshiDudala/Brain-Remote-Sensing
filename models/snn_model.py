#"""
#Spiking Neural Network (SNN) Model for Remote Sensing
#Brain-inspired neural network with temporal dynamics and energy efficiency
#"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import time
import matplotlib.pyplot as plt

class LIFNeuron:
    """Leaky Integrate-and-Fire Neuron Model"""
    
    def __init__(self, threshold=0.5, decay=0.7, reset_voltage=0.0, refractory_period=0):
        self.threshold = threshold
        self.decay = decay
        self.reset_voltage = reset_voltage
        self.refractory_period = refractory_period
        self.membrane_potential = 0.0
        self.last_spike_time = -float('inf')
        self.spike_times = []
    
    def step(self, input_current, dt=1.0, current_time=0):
        """Single time step simulation"""
        # Check refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            return False
        
        # Update membrane potential
        self.membrane_potential = (self.membrane_potential * self.decay + 
                                 input_current * dt)
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.spike_times.append(current_time)
            self.last_spike_time = current_time
            self.membrane_potential = self.reset_voltage
            return True
        
        return False

class SurrogateGradient:
    """Surrogate gradient functions for SNN backpropagation"""
    
    @staticmethod
    def arctan(x, alpha=2.0):
        """Arctan surrogate gradient"""
        return alpha / (1 + alpha**2 * x**2)
    
    @staticmethod
    def sigmoid(x, alpha=1.0):
        """Sigmoid surrogate gradient"""
        return alpha * tf.nn.sigmoid(alpha * x) * (1 - tf.nn.sigmoid(alpha * x))
    
    @staticmethod
    def super_spike(x, beta=10.0):
        """SuperSpike surrogate gradient"""
        return beta / (beta * tf.abs(x) + 1.0)**2

class SpikingLayer(tf.keras.layers.Layer):
    """Custom spiking layer for TensorFlow/Keras"""
    
    def __init__(self, units, threshold=0.5, decay=0.7, dt=1.0, 
                 surrogate_fn='arctan', **kwargs):
        super(SpikingLayer, self).__init__(**kwargs)
        self.units = units
        self.threshold = threshold
        self.decay = decay
        self.dt = dt
        self.surrogate_fn = surrogate_fn
        
        # Initialize surrogate gradient function
        if surrogate_fn == 'arctan':
            self.grad_fn = SurrogateGradient.arctan
        elif surrogate_fn == 'sigmoid':
            self.grad_fn = SurrogateGradient.sigmoid
        elif surrogate_fn == 'super_spike':
            self.grad_fn = SurrogateGradient.super_spike
        else:
            self.grad_fn = SurrogateGradient.arctan
    
    def build(self, input_shape):
        self.membrane_potential = self.add_weight(
            name='membrane_potential',
            shape=(input_shape[-1], self.units),
            initializer='zeros',
            trainable=False
        )
        super(SpikingLayer, self).build(input_shape)
    
    @tf.custom_gradient
    def spike_function(self, membrane_potential):
        """Spike function with surrogate gradient"""
        # Forward pass: Heaviside step function
        spikes = tf.cast(membrane_potential >= self.threshold, tf.float32)
        
        def grad_fn(upstream):
            # Backward pass: surrogate gradient
            return upstream * self.grad_fn(membrane_potential - self.threshold)
        
        return spikes, grad_fn
    
    def call(self, inputs, training=None):
        # Update membrane potential
        self.membrane_potential.assign(
            self.membrane_potential * self.decay + inputs * self.dt
        )
        
        # Generate spikes
        spikes = self.spike_function(self.membrane_potential)
        
        # Reset membrane potential where spikes occurred
        reset_mask = tf.cast(spikes, tf.bool)
        self.membrane_potential.assign(
            tf.where(reset_mask, 0.0, self.membrane_potential)
        )
        
        return spikes

class SpikingConv2D(tf.keras.layers.Layer):
    """Spiking Convolutional Layer"""
    
    def __init__(self, filters, kernel_size, threshold=0.5, decay=0.7, 
                 strides=(1, 1), padding='same', **kwargs):
        super(SpikingConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.decay = decay
        self.strides = strides
        self.padding = padding
        
        # Regular conv layer for weight learning
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False
        )
        
        self.grad_fn = SurrogateGradient.arctan
    
    def build(self, input_shape):
        self.conv.build(input_shape)
        
        # Membrane potential state
        output_shape = self.conv.compute_output_shape(input_shape)
        self.membrane_potential = self.add_weight(
            name='membrane_potential',
            shape=output_shape[1:],
            initializer='zeros',
            trainable=False
        )
        super(SpikingConv2D, self).build(input_shape)
    
    @tf.custom_gradient
    def spike_function(self, membrane_potential):
        spikes = tf.cast(membrane_potential >= self.threshold, tf.float32)
        
        def grad_fn(upstream):
            return upstream * self.grad_fn(membrane_potential - self.threshold)
        
        return spikes, grad_fn
    
    def call(self, inputs, training=None):
        # Convolution
        conv_output = self.conv(inputs)
        
        # Update membrane potential
        self.membrane_potential.assign(
            self.membrane_potential * self.decay + conv_output
        )
        
        # Generate spikes
        spikes = self.spike_function(self.membrane_potential)
        
        # Reset
        reset_mask = tf.cast(spikes, tf.bool)
        self.membrane_potential.assign(
            tf.where(reset_mask, 0.0, self.membrane_potential)
        )
        
        return spikes

class SNNModel:
    """Spiking Neural Network for Remote Sensing Classification"""
    
    def __init__(self, input_shape=(256, 256, 3), num_classes=5, 
                 time_steps=50, dt=1.0):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.dt = dt
        self.model = None
        self.training_history = None
        self.energy_consumption = 0.0
    
    def poisson_encoding(self, images, max_rate=100):
        """Convert images to Poisson spike trains"""
        # Normalize images to [0, 1]
        normalized_images = tf.cast(images, tf.float32) / 255.0
        
        # Generate Poisson spikes
        spike_trains = []
        for t in range(self.time_steps):
            # Generate random numbers
            random_vals = tf.random.uniform(tf.shape(normalized_images))
            
            # Generate spikes based on Poisson process
            spike_prob = normalized_images * max_rate * self.dt / 1000.0
            spikes = tf.cast(random_vals < spike_prob, tf.float32)
            spike_trains.append(spikes)
        
        return tf.stack(spike_trains, axis=1)  # (batch, time, height, width, channels)
    
    def rate_encoding(self, images):
        """Convert images to rate-encoded spikes"""
        # Normalize to [0, 1]
        normalized = tf.cast(images, tf.float32) / 255.0
        
        # Create spike trains with rates proportional to pixel intensity
        spike_trains = []
        for t in range(self.time_steps):
            # Each pixel spikes with probability proportional to its intensity
            random_vals = tf.random.uniform(tf.shape(normalized))
            spikes = tf.cast(random_vals < normalized, tf.float32)
            spike_trains.append(spikes)
        
        return tf.stack(spike_trains, axis=1)
    
    def temporal_encoding(self, images):
        """Convert images to time-to-first-spike encoding"""
        # Normalize and invert (brighter pixels spike earlier)
        normalized = 1.0 - tf.cast(images, tf.float32) / 255.0
        
        # Convert to spike times
        spike_times = normalized * self.time_steps
        
        # Create spike trains
        spike_trains = []
        for t in range(self.time_steps):
            spikes = tf.cast(spike_times <= t + 1, tf.float32) - \
                    tf.cast(spike_times <= t, tf.float32)
            spike_trains.append(spikes)
        
        return tf.stack(spike_trains, axis=1)
    
    def build_snn_architecture(self, encoding='rate', neuron_model='LIF'):
        """Build SNN architecture"""
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoding layer (convert to spikes)
        if encoding == 'rate':
            encoded = layers.Lambda(lambda x: self.rate_encoding(x))(inputs)
        elif encoding == 'poisson':
            encoded = layers.Lambda(lambda x: self.poisson_encoding(x))(inputs)
        else:  # temporal
            encoded = layers.Lambda(lambda x: self.temporal_encoding(x))(inputs)
        
        # Process each time step
        x = encoded
        
        # Spiking convolutional layers
        x = layers.TimeDistributed(
            SpikingConv2D(32, (3, 3), threshold=0.5, decay=0.7)
        )(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        
        x = layers.TimeDistributed(
            SpikingConv2D(64, (3, 3), threshold=0.5, decay=0.7)
        )(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        
        x = layers.TimeDistributed(
            SpikingConv2D(128, (3, 3), threshold=0.5, decay=0.7)
        )(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        
        # Spiking fully connected layers
        x = layers.TimeDistributed(
            SpikingLayer(256, threshold=0.5, decay=0.7)
        )(x)
        
        # Temporal integration and classification
        # Sum spikes over time
        x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
        
        # Final classification layer (non-spiking)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_hybrid_model(self):
        """Build hybrid CNN-SNN model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Traditional CNN feature extraction
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Convert to spike domain
        x = layers.Lambda(lambda x: self.rate_encoding(x * 255))(x)
        
        # Spiking processing
        x = layers.TimeDistributed(
            SpikingConv2D(128, (3, 3), threshold=0.5, decay=0.7)
        )(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        
        # Temporal integration
        x = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(x)
        
        # Classification
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        return models.Model(inputs=inputs, outputs=outputs)
    
    def build_model(self, architecture='snn', **kwargs):
        """Build the model"""
        if architecture == 'snn':
            self.model = self.build_snn_architecture(**kwargs)
        elif architecture == 'hybrid':
            self.model = self.build_hybrid_model()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        return self.model
    
    def compile_model(self, optimizer='adam', learning_rate=0.001):
        """Compile the SNN model"""
        if not self.model:
            raise ValueError("Model not built yet.")
        
        # Use lower learning rate for SNNs
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        # Custom loss that accounts for spike sparsity
        def spike_aware_loss(y_true, y_pred):
            ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            # Add sparsity regularization (encourage fewer spikes)
            sparsity_loss = 0.01 * tf.reduce_mean(y_pred)
            return ce_loss + sparsity_loss
        
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',  # Can use spike_aware_loss
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def train(self, train_data, validation_data=None, epochs=50, 
              batch_size=16, callbacks_list=None):
        """Train the SNN model"""
        if not self.model:
            raise ValueError("Model not built and compiled yet.")
        
        # Default callbacks
        if callbacks_list is None:
            callbacks_list = [
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5)
            ]
        
        start_time = time.time()
        
        self.training_history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"SNN training completed in {training_time:.2f} seconds")
        
        return self.training_history
    
    def predict(self, X, batch_size=16):
        """Make predictions with energy tracking"""
        if not self.model:
            raise ValueError("Model not trained yet.")
        
        start_time = time.time()
        predictions = self.model.predict(X, batch_size=batch_size)
        inference_time = time.time() - start_time
        
        # Estimate energy consumption
        spike_rate = self.estimate_spike_rate(X[:10])  # Sample estimation
        energy_per_inference = self.calculate_energy_consumption(spike_rate)
        total_energy = energy_per_inference * len(X)
        
        return predictions, inference_time, total_energy
    
    def estimate_spike_rate(self, sample_inputs):
        """Estimate average spike rate in the network"""
        # Get intermediate layer outputs (simplified)
        if len(sample_inputs) == 0:
            return 0.1  # Default low spike rate
        
        # Simulate spike rate based on input intensity
        avg_intensity = np.mean(sample_inputs) / 255.0
        # SNNs typically have sparse spiking
        estimated_rate = avg_intensity * 0.3  # 30% max spike rate
        
        return estimated_rate
    
    def calculate_energy_consumption(self, spike_rate):
        """Calculate energy consumption based on spike activity"""
        # Energy model for SNN
        # Base energy per spike event (in microjoules)
        energy_per_spike = 0.1  # μJ per spike
        
        # Get total number of neurons (simplified)
        total_neurons = sum([layer.units if hasattr(layer, 'units') else 
                           np.prod(layer.output_shape[1:]) if hasattr(layer, 'output_shape') else 1000 
                           for layer in self.model.layers])
        
        # Energy per inference
        spikes_per_inference = spike_rate * total_neurons * self.time_steps
        energy_per_inference = spikes_per_inference * energy_per_spike
        
        return energy_per_inference / 1000  # Convert to millijoules
    
    def visualize_spikes(self, sample_input, layer_name=None):
        """Visualize spike patterns in the network"""
        if not self.model:
            raise ValueError("Model not built yet.")
        
        # Create feature extraction model
        if layer_name:
            layer = self.model.get_layer(layer_name)
            extractor = models.Model(inputs=self.model.input, outputs=layer.output)
        else:
            # Use first spiking layer
            spiking_layers = [layer for layer in self.model.layers 
                            if isinstance(layer, (SpikingLayer, SpikingConv2D))]
            if spiking_layers:
                extractor = models.Model(inputs=self.model.input, 
                                       outputs=spiking_layers[0].output)
            else:
                print("No spiking layers found")
                return None
        
        # Get spike patterns
        spike_output = extractor.predict(sample_input[np.newaxis, :])
        
        return spike_output
    
    def plot_spike_raster(self, spike_data, neuron_indices=None, time_window=None):
        """Plot spike raster plot"""
        if spike_data is None:
            print("No spike data available")
            return
        
        # Select subset of neurons and time steps for visualization
        if neuron_indices is None:
            neuron_indices = range(min(20, spike_data.shape[-1]))
        if time_window is None:
            time_window = range(min(self.time_steps, spike_data.shape[1]))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot spikes
        for i, neuron_idx in enumerate(neuron_indices):
            spike_times = []
            if len(spike_data.shape) == 4:  # (batch, time, height, width)
                # For conv layers, take a sample location
                h, w = spike_data.shape[2] // 2, spike_data.shape[3] // 2
                spikes = spike_data[0, :, h, w]
            else:  # (batch, time, neurons)
                spikes = spike_data[0, :, neuron_idx]
            
            for t in time_window:
                if t < len(spikes) and spikes[t] > 0:
                    spike_times.append(t)
            
            if spike_times:
                ax.scatter(spike_times, [i] * len(spike_times), 
                          s=20, c='black', marker='|')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Neuron Index')
        ax.set_title('Spike Raster Plot')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def analyze_temporal_dynamics(self, sample_inputs, num_samples=5):
        """Analyze temporal dynamics of the SNN"""
        results = {
            'spike_rates': [],
            'membrane_potentials': [],
            'temporal_patterns': []
        }
        
        for i in range(min(num_samples, len(sample_inputs))):
            # Get spike patterns
            spike_output = self.visualize_spikes(sample_inputs[i])
            
            if spike_output is not None:
                # Calculate spike rate over time
                if len(spike_output.shape) == 4:
                    spike_rate = np.mean(spike_output[0], axis=(1, 2))
                else:
                    spike_rate = np.mean(spike_output[0], axis=1)
                
                results['spike_rates'].append(spike_rate)
                
                # Analyze temporal patterns
                pattern_analysis = {
                    'max_activity_time': np.argmax(spike_rate),
                    'total_spikes': np.sum(spike_rate),
                    'activity_duration': np.sum(spike_rate > 0.1)
                }
                results['temporal_patterns'].append(pattern_analysis)
        
        return results
    
    def compare_with_cnn_energy(self, cnn_model, sample_data):
        """Compare energy consumption with traditional CNN"""
        # SNN prediction with energy tracking
        snn_pred, snn_time, snn_energy = self.predict(sample_data)
        
        # CNN prediction
        cnn_start = time.time()
        cnn_pred = cnn_model.predict(sample_data)
        cnn_time = time.time() - cnn_start
        
        # Estimate CNN energy (simplified)
        cnn_params = cnn_model.count_params()
        cnn_energy = cnn_params * 0.2 * len(sample_data) / 1000  # Rough estimate
        
        comparison = {
            'snn_energy_mj': snn_energy,
            'cnn_energy_mj': cnn_energy,
            'energy_savings': (cnn_energy - snn_energy) / cnn_energy * 100,
            'snn_time': snn_time,
            'cnn_time': cnn_time,
            'snn_accuracy': None,  # Would need ground truth
            'cnn_accuracy': None
        }
        
        return comparison
    
    def get_model_summary(self):
        """Get detailed model summary"""
        if not self.model:
            raise ValueError("Model not built yet.")
        
        summary_info = {
            'architecture': 'Spiking Neural Network',
            'total_parameters': self.model.count_params(),
            'time_steps': self.time_steps,
            'input_shape': self.input_shape,
            'output_classes': self.num_classes,
            'spiking_layers': len([layer for layer in self.model.layers 
                                 if isinstance(layer, (SpikingLayer, SpikingConv2D))]),
            'estimated_spike_rate': '10-30%',
            'energy_efficiency': 'High (event-driven)'
        }
        
        return summary_info
    
    def save_model(self, filepath):
        """Save the SNN model"""
        if not self.model:
            raise ValueError("Model not built yet.")
        
        # Save model architecture and weights
        self.model.save(filepath)
        
        # Save SNN-specific parameters
        snn_config = {
            'time_steps': self.time_steps,
            'dt': self.dt,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }
        
        config_path = filepath.replace('.h5', '_snn_config.json')
        with open(config_path, 'w') as f:
            import json
            json.dump(snn_config, f)
        
        print(f"SNN model saved to {filepath}")
        print(f"SNN config saved to {config_path}")
    
    def load_model(self, filepath):
        """Load a saved SNN model"""
        # Load model
        self.model = tf.keras.models.load_model(filepath, custom_objects={
            'SpikingLayer': SpikingLayer,
            'SpikingConv2D': SpikingConv2D
        })
        
        # Load SNN config
        config_path = filepath.replace('.h5', '_snn_config.json')
        try:
            with open(config_path, 'r') as f:
                import json
                config = json.load(f)
                self.time_steps = config['time_steps']
                self.dt = config['dt']
                self.input_shape = tuple(config['input_shape'])
                self.num_classes = config['num_classes']
        except FileNotFoundError:
            print("SNN config file not found, using default parameters")
        
        print(f"SNN model loaded from {filepath}")


# Neuromorphic processing utilities
class NeuromorphicProcessor:
    """Utilities for neuromorphic data processing"""
    
    @staticmethod
    def event_based_filtering(image, threshold=0.1):
        """Convert image changes to events"""
        # Simulate event-based vision
        # In real neuromorphic cameras, this would be hardware-generated
        
        # Calculate temporal differences (simulated)
        prev_image = np.roll(image, 1, axis=0)  # Simulate previous frame
        diff = np.abs(image - prev_image)
        
        # Generate events where change exceeds threshold
        events = diff > threshold
        
        return events.astype(np.float32)
    
    @staticmethod
    def address_event_representation(events):
        """Convert events to Address-Event Representation (AER)"""
        # Find event locations
        event_coords = np.where(events)
        
        # Create AER format: (x, y, polarity, timestamp)
        aer_events = []
        for i in range(len(event_coords[0])):
            event = {
                'x': event_coords[1][i],
                'y': event_coords[0][i],
                'polarity': 1,  # Simplified: all positive events
                'timestamp': i  # Sequential timestamp
            }
            aer_events.append(event)
        
        return aer_events
    
    @staticmethod
    def temporal_contrast_normalization(spike_train, time_constant=10):
        """Apply temporal contrast normalization to spike trains"""
        # Exponential moving average for normalization
        normalized = np.zeros_like(spike_train)
        avg = 0.0
        
        for t in range(len(spike_train)):
            avg = avg * np.exp(-1/time_constant) + spike_train[t] * (1 - np.exp(-1/time_constant))
            normalized[t] = spike_train[t] - avg
        
        return normalized


# Example usage and testing
if __name__ == "__main__":
    print("Spiking Neural Network for Remote Sensing - Example Usage")
    
    # Create SNN model
    snn = SNNModel(input_shape=(128, 128, 3), num_classes=5, time_steps=25)
    
    # Build different architectures
    print("\n1. Building pure SNN architecture...")
    model_snn = snn.build_model(architecture='snn', encoding='rate', neuron_model='LIF')
    snn.compile_model(learning_rate=0.0005)
    
    print("Model summary:")
    summary = snn.get_model_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Test spike encoding
    print("\n2. Testing spike encoding methods...")
    test_image = np.random.randint(0, 255, (128, 128, 3)).astype(np.uint8)
    
    # Rate encoding
    rate_spikes = snn.rate_encoding(test_image[np.newaxis, :])
    print(f"Rate encoding output shape: {rate_spikes.shape}")
    print(f"Average spike rate: {np.mean(rate_spikes):.3f}")
    
    # Poisson encoding
    poisson_spikes = snn.poisson_encoding(test_image[np.newaxis, :])
    print(f"Poisson encoding output shape: {poisson_spikes.shape}")
    print(f"Average spike rate: {np.mean(poisson_spikes):.3f}")
    
    # Test energy estimation
    print("\n3. Energy consumption analysis...")
    sample_spike_rate = 0.15  # 15% spike rate
    energy_per_inference = snn.calculate_energy_consumption(sample_spike_rate)
    print(f"Estimated energy per inference: {energy_per_inference:.3f} mJ")
    
    # Traditional CNN comparison
    total_params = model_snn.count_params()
    cnn_energy_estimate = total_params * 0.2 / 1000  # Rough CNN estimate
    energy_savings = (cnn_energy_estimate - energy_per_inference) / cnn_energy_estimate * 100
    print(f"Estimated energy savings vs CNN: {energy_savings:.1f}%")
    
    # Test neuromorphic processing
    print("\n4. Neuromorphic processing features...")
    processor = NeuromorphicProcessor()
    
    # Event-based filtering
    events = processor.event_based_filtering(test_image.astype(np.float32) / 255.0)
    print(f"Generated {np.sum(events)} events from image")
    
    # AER representation
    aer_events = processor.address_event_representation(events)
    print(f"AER format: {len(aer_events)} events")
    if aer_events:
        print(f"First event: {aer_events[0]}")
    
    print("\nSNN model ready for brain-inspired remote sensing!")
    print("Key advantages:")
    print("- Ultra-low power consumption")
    print("- Temporal information processing")
    print("- Event-driven computation")
    print("- Biologically plausible learning")