import pathlib
import qrcode
import cloudinary
import cloudinary.uploader
from PIL import ImageShow
import yaml
          
current_dir = pathlib.Path(__file__).parent.resolve()
secrets = yaml.safe_load(open(f"{current_dir}/secrets.yaml"))
cloudinary.config( 
  cloud_name = secrets.get("cloudinary").get("cloud_name"),
  api_key = secrets.get("cloudinary").get("api_key"),
  api_secret = secrets.get("cloudinary").get("api_secret")
)

def upload_img(image_path):
    # Open the image file
    with open(image_path, 'rb') as file:
        res = cloudinary.uploader.upload(image_path)
        image_url = res["secure_url"]

        # Generate QR code
        qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
        qr.add_data(image_url)
        qr.make(fit=True)
        
        # Create an image from the QR code
        qr_image = qr.make_image(fill_color="black", back_color="white")
        
               # Display the QR code image in a window
        ImageShow.show(qr_image, title='QR Code')
        # Save the QR code image
        # qr_image.save('qr_code.png')
        # print('QR code generated and saved as qr_code.png')