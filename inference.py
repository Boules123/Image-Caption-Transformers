from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from model.model import ImageCaptionModel

def test(test_loader, config):
    model = ImageCaptionModel(config)
    model.load_model(config.checkpoint_dir + '/best_model.pt')
    
    model.eval()
    references = []
    hypotheses = []
    
    for images, captions in test_loader:
        images = images
        captions = captions
        
        generated = model.generate(images, max_len=128)
        
        for i in range(captions.size(0)):
            ref_tokens = captions[i].cpu().tolist()
            gen_tokens = generated[i].cpu().tolist()
            
            # Remove padding and special tokens
            ref_tokens = [t for t in ref_tokens if t > 2]
            gen_tokens = [t for t in gen_tokens if t > 2]
            
            references.append([ref_tokens])  # List of lists for BLEU
            hypotheses.append(gen_tokens)
    
    # Compute BLEU score (corpus-level)
    smoothie = SmoothingFunction().method4
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    
    print(f"Test BLEU Score: {bleu_score:.4f}")
    
    return bleu_score