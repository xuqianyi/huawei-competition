function openModal() {
    document.getElementById('myModal').style.display = "block";
  }

function closeModal() {
    document.getElementById('myModal').style.display = "none";
  }

window.onclick = function(event) {
    if (event.target == document.getElementById('myModal')) {
        closeModal();
    }
}

function submitImage() {
    closeModal();
}

document.getElementById('fileInput').addEventListener('change', function() {
  const file = this.files[0];

  if (file) {
      const reader = new FileReader();

      reader.addEventListener('load', function() {
          document.getElementById('imagePreview').src = this.result;
      });

      reader.readAsDataURL(file);
  }
});
