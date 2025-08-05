#"""
#Vision Transformer (ViT) Model for Remote Sensing
#Attention-based architecture for global context understanding
#"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import time
import math

class PatchEmbedding(tf.keras.layers.Layer):
    """Convert image patches to embeddings"""
    
    def __init__(self, patch_size=16, embed_dim=768, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch extraction using convolution
        self.projection = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid'
        )
    
    def call(self, images):
        # Extract patches and project to embedding space
        batch_size = tf.shape(images)[0]
        
        # Apply convolution to extract patches
        patches = self.projection(images)  # (batch, h_patches, w_patches, embed_dim)
        
        # Reshape to sequence format
        patch_dims = tf.shape(patches)
        patches = tf.reshape(patches, [batch_size, -1, self.embed_dim])
        
        return patches

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.output_dense = layers.Dense(embed_dim)
        
        self.dropout = layers.Dropout(dropout)
        
    def call(self, inputs, training=None, return_attention_weights=False):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Linear projections
        Q = self.query_dense(inputs)  # (batch, seq_len, embed_dim)
        K = self.key_dense(inputs)
        V = self.value_dense(inputs)
        
        # Reshape for multi-head attention
        Q = tf.reshape(Q, [batch_size, seq_len, self.num_heads, self.head_dim])
        K = tf.reshape(K, [batch_size, seq_len, self.num_heads, self.head_dim])
        V = tf.reshape(V, [batch_size, seq_len, self.num_heads, self.head_dim])
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.transpose(V, perm=[0, 2, 1, 3])
        
        # Scaled dot-product attention
        attention_scores = tf.matmul(Q, K, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention to values
        attended_values = tf.matmul(attention_weights, V)
        
        # Reshape back to original format
        attended_values = tf.transpose(attended_values, perm=[0, 2, 1, 3])
        attended_values = tf.reshape(attended_values, [batch_size, seq_len, self.embed_dim])
        
        # Final linear projection
        output = self.output_dense(attended_values)
        
        if return_attention_weights:
            return output, attention_weights
        return output

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer encoder block"""
    
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # MLP block
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout)
        ])
    
    def call(self, inputs, training=None, return_attention_weights=False):
        # Self-attention with residual connection
        if return_attention_weights:
            attention_output, attention_weights = self.attention(
                self.norm1(inputs), training=training, return_attention_weights=True
            )
        else:
            attention_output = self.attention(self.norm1(inputs), training=training)
            attention_weights = None
        
        x = inputs + attention_output
        
        # MLP with residual connection
        mlp_output = self.mlp(self.norm2(x), training=training)
        output = x + mlp_output
        
        if return_attention_weights:
            return output, attention_weights
        return output

class PositionalEncoding(tf.keras.layers.Layer):
    """Add positional encoding to patch embeddings"""
    
    def __init__(self, max_len=1000, embed_dim=768, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        # Create positional encoding matrix
        pos_encoding = self.get_positional_encoding(self.max_len, self.embed_dim)
        self.pos_encoding = self.add_weight(
            name='pos_encoding',
            shape=(self.max_len, self.embed_dim),
            initializer='zeros',
            trainable=False
        )
        self.pos_encoding.assign(pos_encoding)
        
        super(PositionalEncoding, self).build(input_shape)
    
    def get_positional_encoding(self, max_len, embed_dim):
        """Generate sinusoidal positional encodings"""
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(embed_dim)[np.newaxis, :]
        
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        angle_rads = pos * angle_rates
        
        # Apply sin to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        return tf.cast(angle_rads, tf.float32)
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + tf.cast(self.pos_encoding[:seq_len, :], inputs.dtype)

class VisionTransformer(tf.keras.layers.Layer):
    """Complete Vision Transformer architecture"""
    
    def __init__(self, num_classes, patch_size=16, embed_dim=768, 
                 num_heads=12, num_layers=12, mlp_dim=3072, dropout=0.1, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_size, embed_dim)
        
        # Class token
        self.class_token = self.add_weight(
            name='class_token',
            shape=(1, 1, embed_dim),
            initializer='random_normal',
            trainable=True
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim=embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        
        # Layer normalization
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        
        # Classification head
        self.classifier = layers.Dense(num_classes)
        
        self.dropout = layers.Dropout(dropout)
    
    def call(self, inputs, training=None, return_attention_weights=False):
        batch_size = tf.shape(inputs)[0]
        
        # Extract patches and embed
        patches = self.patch_embedding(inputs)
        
        # Add class token
        class_tokens = tf.broadcast_to(self.class_token, [batch_size, 1, self.embed_dim])
        patches = tf.concat([class_tokens, patches], axis=1)
        
        # Add positional encoding
        patches = self.pos_encoding(patches)
        patches = self.dropout(patches, training=training)
        
        # Apply transformer blocks
        attention_weights_list = []
        x = patches
        
        for transformer_block in self.transformer_blocks:
            if return_attention_weights:
                x, attention_weights = transformer_block(
                    x, training=training, return_attention_weights=True
                )
                attention_weights_list.append(attention_weights)
            else:
                x = transformer_block(x, training=training)
        
        # Layer normalization
        x = self.norm(x)
        
        # Classification using class token
        class_token_output = x[:, 0]  # First token is class token
        logits = self.classifier(class_token_output)
        
        if return_attention_weights:
            return logits, attention_weights_list
        return logits

class TransformerModel:
    """Vision Transformer model for remote sensing classification"""
    
    def __init__(self, input_shape=(256, 256, 3), num_classes=5, 
                 model_size='base', patch_size=16):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.model = None
        self.training_history = None
        
        # Model configurations
        self.configs = {
            'tiny': {'embed_dim': 192, 'num_heads': 3, 'num_layers': 12, 'mlp_dim': 768},
            'small': {'embed_dim': 384, 'num_heads': 6, 'num_layers': 12, 'mlp_dim': 1536},
            'base': {'embed_dim': 768, 'num_heads': 12, 'num_layers': 12, 'mlp_dim': 3072},
            'large': {'embed_dim': 1024, 'num_heads': 16, 'num_layers': 24, 'mlp_dim': 4096}
        }
        
        self.config = self.configs.get(model_size, self.configs['base'])
    
    def build_model(self, dropout=0.1, **kwargs):
        """Build Vision Transformer model"""
        
        # Update config with any provided parameters
        config = self.config.copy()
        config.update(kwargs)
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Vision Transformer
        vit = VisionTransformer(
            num_classes=self.num_classes,
            patch_size=self.patch_size,
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            mlp_dim=config['mlp_dim'],
            dropout=dropout
        )
        
        outputs = vit(inputs)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def build_hierarchical_vit(self, window_sizes=[7, 7, 7, 7]):
        """Build hierarchical ViT (similar to Swin Transformer)"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Patch embedding
        x = layers.Conv2D(96, 4, strides=4, padding='same')(inputs)
        x = layers.LayerNormalization()(x)
        
        # Hierarchical stages
        embed_dims = [96, 192, 384, 768]
        
        for i, (embed_dim, window_size) in enumerate(zip(embed_dims, window_sizes)):
            # Patch merging (except first stage)
            if i > 0:
                x = layers.Conv2D(embed_dim, 2, strides=2, padding='same')(x)
                x = layers.LayerNormalization()(x)
            
            # Swin Transformer blocks (simplified)
            for _ in range(2):
                # Window-based self-attention (simplified as regular attention)
                batch_size = tf.shape(x)[0]
                h, w = tf.shape(x)[1], tf.shape(x)[2]
                
                # Reshape for attention
                x_reshaped = tf.reshape(x, [batch_size, h * w, embed_dim])
                
                # Self-attention
                attention_layer = MultiHeadSelfAttention(embed_dim, embed_dim // 32)
                x_attended = attention_layer(x_reshaped)
                
                # Reshape back
                x = tf.reshape(x_attended, [batch_size, h, w, embed_dim])
                
                # MLP
                x = layers.Dense(embed_dim * 4, activation='gelu')(x)
                x = layers.Dense(embed_dim)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Classification
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        return models.Model(inputs=inputs, outputs=outputs)
    
    def build_remote_sensing_vit(self):
        """Build ViT optimized for remote sensing"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Multi-scale patch embedding for remote sensing
        patches_16 = PatchEmbedding(16, 256)(inputs)  # Fine details
        patches_32 = PatchEmbedding(32, 256)(inputs)  # Medium scale
        
        # Spectral attention for remote sensing bands
        if self.input_shape[-1] > 3:  # Multi-spectral
            spectral_attention = layers.Dense(self.input_shape[-1], activation='softmax')
            spectral_weights = spectral_attention(layers.GlobalAveragePooling2D()(inputs))
            spectral_weights = tf.expand_dims(tf.expand_dims(spectral_weights, 1), 1)
            inputs_weighted = inputs * spectral_weights
        else:
            inputs_weighted = inputs
        
        # Regular ViT processing
        vit = VisionTransformer(
            num_classes=self.num_classes,
            patch_size=self.patch_size,
            embed_dim=self.config['embed_dim'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            mlp_dim=self.config['mlp_dim']
        )
        
        outputs = vit(inputs_weighted)
        
        return models.Model(inputs=inputs, outputs=outputs)
    
    def compile_model(self, optimizer='adamw', learning_rate=0.001, 
                     weight_decay=0.05, warmup_steps=1000):
        """Compile Vision Transformer model"""
        if not self.model:
            raise ValueError("Model not built yet.")
        
        # AdamW optimizer with weight decay
        if optimizer == 'adamw':
            # Custom AdamW implementation
            opt = tf.keras.optimizers.experimental.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        # Compile with appropriate loss and metrics
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
    
    def create_learning_rate_schedule(self, warmup_steps=1000, max_lr=0.001):
        """Create learning rate schedule with warmup"""
        def lr_schedule(step):
            step = tf.cast(step, tf.float32)
            warmup_steps_tf = tf.cast(warmup_steps, tf.float32)
            
            # Warmup phase
            warmup_lr = max_lr * step / warmup_steps_tf
            
            # Cosine decay phase
            decay_steps = 10000  # Total training steps
            cosine_lr = max_lr * 0.5 * (1 + tf.cos(np.pi * step / decay_steps))
            
            return tf.where(step < warmup_steps_tf, warmup_lr, cosine_lr)
        
        return lr_schedule
    
    def train(self, train_data, validation_data=None, epochs=100, 
              batch_size=16, callbacks_list=None, use_mixed_precision=True):
        """Train Vision Transformer model"""
        if not self.model:
            raise ValueError("Model not built and compiled yet.")
        
        # Enable mixed precision for faster training
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        # Default callbacks
        if callbacks_list is None:
            callbacks_list = [
                tf.keras.callbacks.EarlyStopping(
                    patience=20, restore_best_weights=True, monitor='val_accuracy'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=10, factor=0.5, min_lr=1e-7
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_vit_model.h5', save_best_only=True, monitor='val_accuracy'
                )
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
        print(f"ViT training completed in {training_time:.2f} seconds")
        
        return self.training_history
    
    def predict(self, X, batch_size=16, return_attention=False):
        """Make predictions with optional attention visualization"""
        if not self.model:
            raise ValueError("Model not trained yet.")
        
        start_time = time.time()
        
        if return_attention:
            # Create model that returns attention weights
            vit_layer = None
            for layer in self.model.layers:
                if isinstance(layer, VisionTransformer):
                    vit_layer = layer
                    break
            
            if vit_layer:
                # Get attention weights
                inputs = self.model.input
                outputs, attention_weights = vit_layer(inputs, return_attention_weights=True)
                attention_model = models.Model(inputs=inputs, 
                                             outputs=[outputs, attention_weights])
                predictions, attention = attention_model.predict(X, batch_size=batch_size)
            else:
                predictions = self.model.predict(X, batch_size=batch_size)
                attention = None
        else:
            predictions = self.model.predict(X, batch_size=batch_size)
            attention = None
        
        inference_time = time.time() - start_time
        
        if return_attention:
            return predictions, attention, inference_time
        return predictions, inference_time
    
    def visualize_attention(self, image, layer_idx=-1, head_idx=0):
        """Visualize attention maps"""
        if not self.model:
            raise ValueError("Model not built yet.")
        
        # Get attention weights
        predictions, attention_weights, _ = self.predict(
            image[np.newaxis, :], return_attention=True
        )
        
        if attention_weights is None:
            print("No attention weights available")
            return None
        
        # Select specific layer and head
        attention = attention_weights[layer_idx][0, head_idx]  # (seq_len, seq_len)
        
        # Extract attention to class token (first token)
        class_attention = attention[0, 1:]  # Exclude class token self-attention
        
        # Reshape to spatial dimensions
        num_patches = int(np.sqrt(len(class_attention)))
        attention_map = class_attention.reshape(num_patches, num_patches)
        
        # Resize to original image size
        attention_resized = tf.image.resize(
            attention_map[np.newaxis, :, :, np.newaxis],
            [image.shape[0], image.shape[1]]
        )[0, :, :, 0]
        
        return attention_resized.numpy()
    
    def analyze_patch_importance(self, images, num_samples=5):
        """Analyze which patches are most important for classification"""
        results = {
            'patch_importance': [],
            'spatial_attention': [],
            'classification_confidence': []
        }
        
        for i in range(min(num_samples, len(images))):
            # Get predictions and attention
            pred, attention, _ = self.predict(
                images[i:i+1], return_attention=True
            )
            
            if attention is not None:
                # Average attention across heads and layers
                avg_attention = np.mean([np.mean(layer_att[0], axis=0) 
                                       for layer_att in attention], axis=0)
                
                # Class token attention (importance of each patch)
                patch_importance = avg_attention[0, 1:]  # Exclude class token
                
                results['patch_importance'].append(patch_importance)
                results['classification_confidence'].append(np.max(pred[0]))
                
                # Spatial attention map
                num_patches = int(np.sqrt(len(patch_importance)))
                spatial_map = patch_importance.reshape(num_patches, num_patches)
                results['spatial_attention'].append(spatial_map)
        
        return results
    
    def compare_architectures(self, test_data):
        """Compare different ViT architectures"""
        architectures = ['tiny', 'small', 'base']
        comparison_results = {}
        
        for arch in architectures:
            print(f"Testing {arch} architecture...")
            
            # Create model
            test_model = TransformerModel(
                self.input_shape, self.num_classes, arch, self.patch_size
            )
            model = test_model.build_model()
            test_model.compile_model()
            
            # Benchmark inference speed
            start_time = time.time()
            predictions = model.predict(test_data[:10])  # Small sample
            inference_time = (time.time() - start_time) / 10 * 1000  # ms per image
            
            # Get model stats
            params = model.count_params()
            
            comparison_results[arch] = {
                'parameters': params,
                'inference_time_ms': inference_time,
                'memory_mb': params * 4 / (1024 * 1024),  # Rough estimate
                'config': test_model.config
            }
        
        return comparison_results
    
    def get_feature_representations(self, images, layer_name=None):
        """Extract feature representations from transformer layers"""
        if not self.model:
            raise ValueError("Model not built yet.")
        
        # Find transformer blocks
        transformer_layers = []
        for layer in self.model.layers:
            if isinstance(layer, VisionTransformer):
                # Get transformer blocks from ViT layer
                for block in layer.transformer_blocks:
                    transformer_layers.append(block)
                break
        
        if not transformer_layers:
            print("No transformer layers found")
            return None
        
        # Create feature extraction model
        vit_layer = None
        for layer in self.model.layers:
            if isinstance(layer, VisionTransformer):
                vit_layer = layer
                break
        
        if vit_layer is None:
            return None
        
        # Extract intermediate features
        inputs = self.model.input
        
        # Get patch embeddings
        patches = vit_layer.patch_embedding(inputs)
        class_tokens = tf.broadcast_to(
            vit_layer.class_token, 
            [tf.shape(inputs)[0], 1, vit_layer.embed_dim]
        )
        patches_with_cls = tf.concat([class_tokens, patches], axis=1)
        patches_with_pos = vit_layer.pos_encoding(patches_with_cls)
        
        # Extract features from different layers
        feature_extractor = models.Model(
            inputs=inputs,
            outputs=patches_with_pos
        )
        
        features = feature_extractor.predict(images)
        return features
    
    def save_model(self, filepath):
        """Save Vision Transformer model"""
        if not self.model:
            raise ValueError("Model not built yet.")
        
        self.model.save(filepath)
        
        # Save configuration
        config_data = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'patch_size': self.patch_size,
            'model_config': self.config
        }
        
        config_path = filepath.replace('.h5', '_vit_config.json')
        with open(config_path, 'w') as f:
            import json
            json.dump(config_data, f, default=str)
        
        print(f"ViT model saved to {filepath}")
        print(f"Configuration saved to {config_path}")
    
    def load_model(self, filepath):
        """Load Vision Transformer model"""
        # Custom objects for loading
        custom_objects = {
            'VisionTransformer': VisionTransformer,
            'PatchEmbedding': PatchEmbedding,
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'TransformerBlock': TransformerBlock,
            'PositionalEncoding': PositionalEncoding
        }
        
        self.model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
        
        # Load configuration
        config_path = filepath.replace('.h5', '_vit_config.json')
        try:
            with open(config_path, 'r') as f:
                import json
                config_data = json.load(f)
                self.input_shape = tuple(config_data['input_shape'])
                self.num_classes = config_data['num_classes']
                self.patch_size = config_data['patch_size']
                self.config = config_data['model_config']
        except FileNotFoundError:
            print("Configuration file not found, using default settings")
        
        print(f"ViT model loaded from {filepath}")
    
    def get_model_summary(self):
        """Get detailed model summary"""
        if not self.model:
            raise ValueError("Model not built yet.")
        
        summary_info = {
            'architecture': 'Vision Transformer',
            'total_parameters': self.model.count_params(),
            'patch_size': self.patch_size,
            'embed_dim': self.config['embed_dim'],
            'num_heads': self.config['num_heads'],
            'num_layers': self.config['num_layers'],
            'mlp_dim': self.config['mlp_dim'],
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'attention_mechanism': 'Multi-head Self-attention',
            'global_context': 'Yes (all patches)',
            'computational_complexity': 'O(n²) where n is number of patches'
        }
        
        return summary_info


# Utility functions for remote sensing ViT
def create_multi_scale_patches(image, patch_sizes=[8, 16, 32]):
    """Create patches at multiple scales for remote sensing"""
    patches_dict = {}
    
    for patch_size in patch_sizes:
        patches = tf.image.extract_patches(
            images=image[np.newaxis, :],
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        
        patches_dict[f'patches_{patch_size}'] = patches[0]
    
    return patches_dict

def spectral_attention_weights(image_bands):
    """Calculate attention weights for spectral bands"""
    # Simple spectral attention based on variance
    band_variances = [np.var(band) for band in tf.unstack(image_bands, axis=-1)]
    
    # Normalize to get attention weights
    attention_weights = tf.nn.softmax(tf.constant(band_variances, dtype=tf.float32))
    
    return attention_weights

def visualize_transformer_attention(attention_map, original_image, save_path=None):
    """Visualize transformer attention overlaid on original image"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    im1 = axes[1].imshow(attention_map, cmap='hot', interpolation='bilinear')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Overlay
    axes[2].imshow(original_image)
    axes[2].imshow(attention_map, alpha=0.6, cmap='hot', interpolation='bilinear')
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Example usage and testing
if __name__ == "__main__":
    print("Vision Transformer for Remote Sensing - Example Usage")
    
    # Create ViT model
    vit = TransformerModel(
        input_shape=(224, 224, 3), 
        num_classes=6, 
        model_size='base', 
        patch_size=16
    )
    
    # Build different architectures
    print("\n1. Building standard ViT...")
    model_standard = vit.build_model(dropout=0.1)
    vit.compile_model(learning_rate=0.001)
    
    print("Model summary:")
    summary = vit.get_model_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Test multi-scale patches
    print("\n2. Testing multi-scale patch extraction...")
    test_image = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    multi_patches = create_multi_scale_patches(test_image)
    
    for patch_name, patches in multi_patches.items():
        print(f"{patch_name} shape: {patches.shape}")
    
    # Test spectral attention
    print("\n3. Testing spectral attention...")
    if len(test_image.shape) == 3 and test_image.shape[-1] >= 3:
        spectral_weights = spectral_attention_weights(test_image.astype(np.float32))
        print(f"Spectral attention weights: {spectral_weights.numpy()}")
    
    # Architecture comparison
    print("\n4. Comparing ViT architectures...")
    dummy_data = np.random.random((5, 224, 224, 3))
    comparison = vit.compare_architectures(dummy_data)
    
    print("Architecture Comparison:")
    for arch, metrics in comparison.items():
        print(f"\n{arch.upper()}:")
        print(f"  Parameters: {metrics['parameters']:,}")
        print(f"  Inference time: {metrics['inference_time_ms']:.2f} ms")
        print(f"  Memory usage: {metrics['memory_mb']:.1f} MB")
    
    # Test attention visualization
    print("\n5. Testing attention mechanisms...")
    
    # Calculate number of patches
    H, W = test_image.shape[:2]
    patch_size = vit.patch_size
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_h * num_patches_w
    
    print(f"Image size: {H}x{W}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Number of patches: {total_patches} ({num_patches_h}x{num_patches_w})")
    print(f"Sequence length (with class token): {total_patches + 1}")
    
    # Attention complexity
    attention_complexity = (total_patches + 1) ** 2 * vit.config['num_heads'] * vit.config['num_layers']
    print(f"Attention computation complexity: {attention_complexity:,} operations")
    
    print("\nVision Transformer ready for remote sensing applications!")
    print("Key advantages:")
    print("- Global context understanding")
    print("- Self-attention mechanisms")
    print("- Scalable to high-resolution images")
    print("- Interpretable attention patterns")
    print("- State-of-the-art performance on many tasks")