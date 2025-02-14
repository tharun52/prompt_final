from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from PIL import Image
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Replace with a strong secret key

# Configure upload folder and allowed file extensions
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class LightImageEvaluator:
    def __init__(self):
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.transform = Compose([
            Resize((224, 224), antialias=True),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.process_size = (160, 160)

    def structural_similarity(self, original, generated):
        orig_small = cv2.resize(np.array(original), self.process_size, interpolation=cv2.INTER_AREA)
        gen_small = cv2.resize(np.array(generated), self.process_size, interpolation=cv2.INTER_AREA)
        
        orig_gray = cv2.cvtColor(orig_small, cv2.COLOR_RGB2GRAY)
        gen_gray = cv2.cvtColor(gen_small, cv2.COLOR_RGB2GRAY)
        
        return ssim(orig_gray, gen_gray, data_range=255)

    def feature_similarity(self, original, generated):
        try:
            orig_input = self.transform(original).unsqueeze(0)
            gen_input = self.transform(generated).unsqueeze(0)
            
            with torch.no_grad():
                orig_features = self.model(orig_input)
                gen_features = self.model(gen_input)
                
                similarity = F.cosine_similarity(orig_features, gen_features).item()
            
            del orig_input, gen_input, orig_features, gen_features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return similarity
        except RuntimeError as e:
            print(f"Warning: Feature extraction failed: {e}")
            return 0.0

    def compute_comprehensive_score(self, original, generated):
        try:
            ssim_score = self.structural_similarity(original, generated)
            feature_score = self.feature_similarity(original, generated)
            
            comprehensive_score = (0.5 * ssim_score) + (0.5 * feature_score)
            
            return {
                'comprehensive_score': comprehensive_score,
                'metrics': {
                    'SSIM': ssim_score,
                    'Feature Similarity': feature_score
                }
            }
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {
                'comprehensive_score': 0.0,
                'metrics': {
                    'SSIM': 0.0,
                    'Feature Similarity': 0.0
                }
            }

def evaluate_recreation(original_path, generated_path, max_size=800):
    try:
        original = Image.open(original_path).convert('RGB')
        generated = Image.open(generated_path).convert('RGB')
        
        if max(original.size) > max_size:
            ratio = max_size / max(original.size)
            new_size = tuple(int(dim * ratio) for dim in original.size)
            original = original.resize(new_size, Image.LANCZOS)
        
        if max(generated.size) > max_size:
            ratio = max_size / max(generated.size)
            new_size = tuple(int(dim * ratio) for dim in generated.size)
            generated = generated.resize(new_size, Image.LANCZOS)
        
        if original.size != generated.size:
            generated = generated.resize(original.size, Image.LANCZOS)
        
        evaluator = LightImageEvaluator()
        result = evaluator.compute_comprehensive_score(original, generated)
        return int(result['comprehensive_score'] * 10000)/100

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

@app.route('/')
def index():
    images = [
        {'id': '1', 'filename': 'image1.jpg'},
        {'id': '2', 'filename': 'image2.jpg'},
        {'id': '3', 'filename': 'image3.jpg'}
    ]
    scores = session.get('scores', {})
    average_score = None
    if len(scores) == 3:
        average_score = sum(scores.values()) / 3

    return render_template('index.html', images=images, scores=scores, average_score=average_score)

@app.route('/play/<image_id>', methods=['GET', 'POST'])
def play(image_id):
    mapping = {
        '1': 'image1.jpg',
        '2': 'image2.jpg',
        '3': 'image3.jpg'
    }
    if image_id not in mapping:
        flash("Invalid image selected.")
        return redirect(url_for('index'))

    original_image = mapping[image_id]
    original_image_path = os.path.join('static', 'images', original_image)

    if request.method == 'POST':
        if 'uploaded_image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['uploaded_image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(uploaded_path)

            score = evaluate_recreation(original_image_path, uploaded_path)
            
            scores = session.get('scores', {})
            scores[image_id] = score
            session['scores'] = scores

            average_score = None
            if len(scores) == 3:
                average_score = sum(scores.values()) / 3

            return render_template('result.html',
                                   original_image=url_for('static', filename='images/' + original_image),
                                   uploaded_image=url_for('static', filename='uploads/' + filename),
                                   score=score,
                                   average_score=average_score)
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)

    return render_template('play.html', image_id=image_id, original_image=url_for('static', filename='images/' + original_image))

if __name__ == '__main__':
    app.run(debug=True)
