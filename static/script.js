function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function(){
        const previewContainer = document.getElementById('image-preview-container');
        const preview = document.getElementById('image-preview');
        
        preview.src = reader.result;
        previewContainer.style.display = 'block';
        
        // Hide the previous result if a new image is selected
        const resultContainer = document.querySelector('.prediction-result');
        if (resultContainer) {
            resultContainer.style.display = 'none';
        }
    };
    reader.readAsDataURL(event.target.files[0]);
}

// Optional: Add a listener to the form to show a loading state
document.getElementById('upload-form').addEventListener('submit', function() {
    const button = this.querySelector('.button-predict');
    button.textContent = '正在分析...';
    button.disabled = true;
}); 