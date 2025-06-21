import torch
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize the model
embed_size = 300
vocab_size = tokenizer.vocab_size  # Using tokenizer's vocab size
attention_dim = 256
encoder_dim = 2048
decoder_dim = 512
drop_prob = 0.3

model = EncoderDecoder(
    embed_size=embed_size,
    attention_dim=attention_dim,
    encoder_dim=encoder_dim,
    decoder_dim=decoder_dim,
    drop_prob=drop_prob
).to(device)

# Load the pre-trained model weights
model.load_state_dict(torch.load("/kaggle/working/model_weights.pth", map_location=device))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_caption_beam_search(image_path, model, tokenizer, beam_size=5, max_len=50):
    """
    Generate caption using beam search
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Encode the image
    features = model.encoder(image_tensor)  # Shape: (1, num_pixels, encoder_dim)
    
    # Expand features for beam search
    features = features.expand(beam_size, features.size(1), features.size(2))  # (beam_size, num_pixels, encoder_dim)
    
    # Initialize beam search variables
    beam_size = beam_size
    vocab_size = tokenizer.vocab_size
    
    # Initialize sequences and scores
    sequences = torch.tensor([[tokenizer.cls_token_id]], device=device).expand(beam_size, 1)  # (beam_size, 1)
    scores = torch.zeros(beam_size, device=device)  # (beam_size,)
    
    # Initialize hidden states for each beam
    h, c = model.decoder.init_hidden_state(features)  # Both (beam_size, decoder_dim)
    
    # Store completed sequences
    completed_sequences = []
    completed_scores = []
    
    # Beam search loop
    for step in range(max_len):
        # Get embeddings for current sequences
        if step == 0:
            # First step: use CLS token
            embeds = model.decoder.embedding(sequences[:, -1:])  # (beam_size, 1, embed_size)
        else:
            embeds = model.decoder.embedding(sequences[:, -1:])  # (beam_size, 1, embed_size)
        
        # Forward pass through decoder
        alpha, context = model.decoder.attention(features, h)  # alpha: (beam_size, num_pixels), context: (beam_size, encoder_dim)
        
        lstm_input = torch.cat((embeds.squeeze(1), context), dim=1)  # (beam_size, embed_size + encoder_dim)
        h, c = model.decoder.lstm_cell(lstm_input, (h, c))
        
        output = model.decoder.fcn(model.decoder.drop(h))  # (beam_size, vocab_size)
        log_probs = F.log_softmax(output, dim=1)  # (beam_size, vocab_size)
        
        # Calculate scores for all possible next words
        if step == 0:
            # First step: all beams are identical, so only consider first beam
            scores = scores[0].unsqueeze(0) + log_probs[0]  # (vocab_size,)
            top_scores, top_words = scores.topk(beam_size, dim=0)
            
            # Update sequences
            sequences = torch.cat([sequences[0].unsqueeze(0).expand(beam_size, -1), 
                                 top_words.unsqueeze(1)], dim=1)  # (beam_size, 2)
            scores = top_scores
            
            # Expand hidden states
            h = h[0].unsqueeze(0).expand(beam_size, -1)
            c = c[0].unsqueeze(0).expand(beam_size, -1)
        else:
            # Subsequent steps: consider all beams
            scores = scores.unsqueeze(1) + log_probs  # (beam_size, vocab_size)
            scores = scores.view(-1)  # (beam_size * vocab_size,)
            
            # Get top beam_size scores
            top_scores, top_indices = scores.topk(beam_size, dim=0)
            
            # Convert flat indices to beam and word indices
            beam_indices = top_indices // vocab_size
            word_indices = top_indices % vocab_size
            
            # Update sequences
            new_sequences = []
            new_h = []
            new_c = []
            
            for i in range(beam_size):
                beam_idx = beam_indices[i]
                word_idx = word_indices[i]
                
                # Append new word to sequence
                new_seq = torch.cat([sequences[beam_idx], word_idx.unsqueeze(0)])
                new_sequences.append(new_seq)
                
                # Copy hidden states
                new_h.append(h[beam_idx])
                new_c.append(c[beam_idx])
            
            sequences = torch.stack([seq for seq in new_sequences])  # (beam_size, seq_len)
            h = torch.stack(new_h)  # (beam_size, decoder_dim)
            c = torch.stack(new_c)  # (beam_size, decoder_dim)
            scores = top_scores
        
        # Check for completed sequences (SEP token)
        sep_positions = (sequences[:, -1] == tokenizer.sep_token_id)
        
        if sep_positions.any():
            # Store completed sequences
            for i in range(beam_size):
                if sep_positions[i]:
                    completed_sequences.append(sequences[i].cpu().numpy())
                    completed_scores.append(scores[i].item())
            
            # Remove completed sequences from active beams
            if sep_positions.all():
                break
                
            # Continue with non-completed sequences
            active_beams = ~sep_positions
            if active_beams.sum() > 0:
                sequences = sequences[active_beams]
                scores = scores[active_beams]
                h = h[active_beams]
                c = c[active_beams]
                features = features[active_beams]
                beam_size = active_beams.sum().item()
    
    # If no sequences completed with SEP token, use the current sequences
    if not completed_sequences:
        for i in range(len(sequences)):
            completed_sequences.append(sequences[i].cpu().numpy())
            completed_scores.append(scores[i].item())
    
    # Select best sequence based on normalized score (to account for length)
    if completed_sequences:
        # Normalize scores by sequence length
        normalized_scores = []
        for seq, score in zip(completed_sequences, completed_scores):
            length = len(seq) - 1  # Subtract 1 for CLS token
            normalized_scores.append(score / max(length, 1))
        
        best_idx = np.argmax(normalized_scores)
        best_sequence = completed_sequences[best_idx]
        
        # Convert to caption
        caption = tokenizer.decode(best_sequence, skip_special_tokens=True)
        
        return image, caption
    else:
        return image, "Unable to generate caption"



# Test the inference with beam search and display the image with caption
image_path = "/kaggle/input/testimage/caninecottages_guides_1725979524538-Untitled-1.webp"

# Generate caption using beam search
image, caption = predict_caption_beam_search(image_path, model, tokenizer, beam_size=5)

# Display the image with the generated caption
plt.imshow(image)
plt.axis("off")
plt.show()

print(f"Caption: {caption}")