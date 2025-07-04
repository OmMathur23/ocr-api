from flask import Flask, request, jsonify,render_template
import requests
import re
import re, pytesseract, cv2, numpy as np
from PIL import Image
import ftfy
from rapidfuzz import fuzz
import sqlite3
import pdfplumber
import os
import traceback
import datetime
import json
#import easyocr
from pdf2image import convert_from_bytes
import atexit
import json
import logging.config
import logging.handlers
import pathlib
from paddleocr import PaddleOCR
#import spacy
#from spacy.training.example import Example
#from spacy.util import minibatch
import random
import os
import mysql.connector
from gevent.pywsgi import WSGIServer
from dateutil import parser
import datetime
from werkzeug.datastructures import FileStorage
from io import BytesIO
from PIL import Image
from flask_cors import CORS
os.environ["FLAGS_bvar_disable"] = "1"
import ftfy 
import fitz
import io

from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity,JWTManager
from memory_profiler import profile
from difflib import SequenceMatcher

def connectDB(dictionary=False):
    mydb = mysql.connector.connect(
        host="localhost",
        port=3306,  # Separate the port number from the host
        user="ocr",
        password="ocr@420",
        database="ocr"
    )
    if dictionary:
      cursor = mydb.cursor(dictionary=True)  # Enable dictionary cursor
      return mydb, cursor
    return mydb, mydb.cursor()






# conn, cursor = connectDB()



# cursor.execute("SELECT * FROM companydetails ")
# data = cursor.fetchall()
# print(data,"companydetails")


# cursor.execute("SELECT * FROM ocrdetails ")

# data = cursor.fetchall()
# print(data,"ocrdetails")




def setup_logging():
    config_file = pathlib.Path("config/log_config.json")

    # Ensure logs directory exists
    log_dir = pathlib.Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)  # Create logs/ if missing

    with open(config_file) as f_in:
        config = json.load(f_in)

    logging.config.dictConfig(config)

    queue_handler = None
    for handler in logging.root.handlers:
        if handler.get_name() == "queue_handler":
            queue_handler = handler
            break

    if queue_handler is not None and hasattr(queue_handler, 'listener'):
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)
        
    
setup_logging()

logger = logging.getLogger(__name__)
            

            
        
        
        
        

app = Flask(__name__)



#app.config["JWT_SECRET_KEY"] = "Om@123"  # Replace with a strong secret key

jwt = JWTManager(app)

cors = CORS(app)
def convert_to_ddmmyyyy(date_str):
    try:
        parsed_date = parser.parse(date_str)
        return parsed_date.strftime("%d-%m-%Y")
    except ValueError:
        return "Invalid date format"
    

    
@app.route('/', methods=['GET'])
def ocr():
    logger.info("Received request at '/'", extra={"route": "/", "method": "GET"})
    return jsonify({"message": "Welcome to OCR API!"})


@app.route('/get_token', methods=['POST'])
def get_token():
    data = request.get_json()
    company_id=data.get("company_id")
    email=data.get("email")
    password=data.get("password")
    
    conn, cursor = connectDB()
    cursor.execute(" select password  from companydetails where  company_id=%s and email =%s ",(company_id,email))
    data1 =cursor.fetchall()
    try:
        dbpassword=data1[0][0]
        if dbpassword==password:
            
    
            expires = datetime.timedelta(minutes=60)
            access_token = create_access_token(identity=company_id, expires_delta=expires) 
            return jsonify({"token":access_token,"success":True})
        
    except Exception as e:
        traceback.print_exc()
        pass
    
    
    return jsonify({"token":"","success":False})
    
def check_auth(token,file):
    
    try:
        conn, cursor = connectDB()
        
        cursor.execute(f" select {file},credits_left,is_negative_credit_allowed  from companydetails where  company_id=%s ",(token,))
        document_access =cursor.fetchall()
        if  document_access:
            if document_access[0][0]==1:
                if int(document_access[0][1])<0:
                    if int(document_access[0][2])==1:
                        
                        return True," access granted"
                    else:
                        
                        return False,f"credit not avilable"
                        
                        
                    
                
                return True," access granted"
            else:
                 return False,f" {file} not provided "
                
            
        else:
            
            return False,"unauthorized access"
    except Exception as e:
        traceback.print_exc()
        
        return False ,"unauthorized access"
    
    
#def retrain_spacy_ner(train_data, model_path="/home/meon/ocr/custom_pan_ner", iterations=100):
    try:
        if os.path.exists(model_path):
            print(f"📥 Loading existing model from '{model_path}'...")
            nlp = spacy.load(model_path)
        else:
            print(f"⚠ Model '{model_path}' not found! Creating a new blank model.")
            nlp = spacy.blank("en")

        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")

        # Register all labels
        for _, annotations in train_data:
            for start, end, label in annotations["entities"]:
                ner.add_label(label)

        # Disable other pipes
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):
            if os.path.exists(model_path):
                optimizer = nlp.resume_training()
            else:
                optimizer = nlp.begin_training()
        
            
            for i in range(iterations):
                random.shuffle(train_data)
                losses = {}
                batches = minibatch(train_data, size=2)

                for batch in batches:
                    for text, annotations in batch:
                        doc = nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        nlp.update([example], losses=losses)

                print(f"Iteration {i+1}/{iterations} - Losses: {losses}")

        nlp.to_disk(model_path)
        print(f"✅ Model retrained and saved to '{model_path}'")

        return f"✅ Model retrained and saved to '{model_path}'"
    except Exception as e:
        return str(e)

def normalize_dob_format(dob):
    """Ensures DOB follows the DD-MM-YYYY format."""
    if re.match(r'\d{2}/\d{2}/\d{4}', dob):
        return dob.replace('/', '-')
    return dob


def correct_pan_number(pan):
    """Corrects OCR errors in the PAN number format (ABCDE1234F)."""
    if not pan or len(pan) != 10:
        return pan  # Return as is if format is incorrect

    # Convert to a list to modify characters at specific positions
    corrected_pan = list(pan)

    # Only replace 'S' with '5' in positions 5-8
    for i in range(5, 9):
        if corrected_pan[i] == 'S':  
            corrected_pan[i] = '5'
        if corrected_pan[i] == 'S':  
            corrected_pan[i] = '5' 
        if corrected_pan[i]=='A':
             corrected_pan[i] = '4'
        if corrected_pan[i]=='T':
            corrected_pan[i]='7'  

    return "".join(corrected_pan)


def string_matching_percentage(str1, str2):
    """
    Compares two strings and returns a match percentage.
    Handles cases where either string is None.
    """
    str1 = str1.lower().strip() if str1 else ""
    str2 = str2.lower().strip() if str2 else ""
    return fuzz.ratio(str1, str2)



def extract_financial_data(text):
    """
    Extracts financial-related fields (Client Name, MICR Code, IFSC Code, Account Number) from OCR text.
    Includes debugging statements to validate and debug the extracted text.
    """
    details = {
        "Client Name": None,
        "MICR Code": None,
        "IFSC Code": None,
        "Account Number": None
    }

    lines = text.split("\n")

    print("OCR Lines:", lines)

    processed_lines = [line.strip() for line in lines if line.strip()]
    print("Processed Lines:", processed_lines)

    for line in processed_lines:
        name_match = re.search(r'Name[:\s]+(.+)', line, re.IGNORECASE)
        if name_match:
            potential_name = clean_text(name_match.group(1).strip())
            if not any(word in potential_name.lower() for word in ['address', 'w/o', 'd/o', 's/o', 'pincode', 'holder', 'nominee']):
                details["Client Name"] = potential_name
                print("Extracted Client Name:", potential_name)
                break

    if not details["Client Name"]:
        for line in processed_lines:
            if len(line.split()) > 1 and not re.search(r'(bank|branch|account|code|address|holder|nominee|statement)', line, re.IGNORECASE):
                details["Client Name"] = clean_text(line)
                print("Fallback Extracted Client Name:", details["Client Name"])
                break

    for line in processed_lines:
        ifsc_match = re.search(r'[A-Z]{4}0[A-Z0-9]{6}', line)
        if ifsc_match:
            details["IFSC Code"] = clean_text(ifsc_match.group(0))
            print("Extracted IFSC Code:", details["IFSC Code"])
            break

    for line in processed_lines:
        micr_match = re.search(r'\b\d{9}\b', line)
        if micr_match:
            details["MICR Code"] = clean_text(micr_match.group(0))
            print("Extracted MICR Code:", details["MICR Code"])
            break

    for line in processed_lines:
        account_match = re.search(r'\b\d{10,16}\b', line)
        if account_match:
            details["Account Number"] = clean_text(account_match.group(0))
            print("Extracted Account Number:", details["Account Number"])
            break

    print("Extracted Financial Details:", details)

    return details



def normalize_dob_format(ocr_dob):
    """
    Normalize the DOB format from OCR (DD/MM/YYYY) to database format (DD-MM-YYYY).
    """
    if "/" in ocr_dob:
        return ocr_dob.replace("/", "-").strip("-")
    return ocr_dob


def clean_text(text):
    """
    Cleans up non-standard characters, fixes encoding issues, and trims whitespace.
    """
    text = ftfy.fix_text(text)
    text = re.sub(r'\(cid:\d+\)', '', text)
    return text.strip()


# def extract_pan_data(text):
#     """
#     Extracts PAN card details dynamically from OCR text lines, including Name, Father's Name, DOB, and PAN Number.
#     """
#     extracted_data = {
#         'ocr_name': '',
#         'ocr_father_name': '',
#         'ocr_dob': '',
#         'ocr_pan_number': ''
#     }

#     lines = [line.strip() for line in text.split('\n') if line.strip()]
#     print("OCR Lines:", lines)

#     for i, line in enumerate(lines):
#         clean_line = re.sub(r'[^A-Za-z0-9 /-]+', '', line)  

#         if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', clean_line):
#             extracted_data['ocr_pan_number'] = clean_line
#             continue
#         elif re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', clean_line):  
#             match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', clean_line)
#             if match:
#                 extracted_data['ocr_pan_number'] = match.group(0)
#             continue

#         if re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', line):
#             dob_match = re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', line)
#             if dob_match:
#                 extracted_data['ocr_dob'] = dob_match.group(0).replace('/', '-')  
#             continue

#         if (
#             line.isupper()
#             and not extracted_data['ocr_name']  
#             and not any(keyword in line for keyword in ["GOVT.", "INCOME TAX", "PERMANENT ACCOUNT NUMBER", "CARD", "DATE OF BIRTH", "FATHER"])
#             and len(clean_line) > 2 
#         ):
#             extracted_data['ocr_name'] = clean_line
#             if i + 1 < len(lines) and lines[i + 1].isupper():
#                 extracted_data['ocr_father_name'] = lines[i + 1].strip()
#             continue

#         if "FATHER" in line.upper() and i + 1 < len(lines):
#             extracted_data['ocr_father_name'] = lines[i + 1].strip()
#             continue

#     extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', extracted_data['ocr_name']).strip()
#     extracted_data['ocr_father_name'] = re.sub(r'[^A-Za-z ]+', '', extracted_data['ocr_father_name']).strip()
#     extracted_data['ocr_dob'] = normalize_dob_format(extracted_data['ocr_dob'])
#     extracted_data['ocr_pan_number'] = re.sub(r'[^A-Z0-9]+', '', extracted_data['ocr_pan_number']).strip()
    

 
def convert_pdf_to_image(file,path):
    
    pdf_bytes = file.read()  # Read PDF as bytes
    
    images = convert_from_bytes(pdf_bytes, dpi=300)
    
    if len(images) == 1:
        images[0].save(path)
        
        return True
    
    elif len(images) == 2:
                # Merge two pages into one image
            img1, img2 = images[0], images[1]
            new_width = max(img1.width, img2.width)
            new_height = img1.height + img2.height

            merged_img = Image.new("RGB", (new_width, new_height), (255, 255, 255))
            merged_img.paste(img1, (0, 0))
            merged_img.paste(img2, (0, img1.height))
            merged_img.save(path)
            
            return True
    
    return False
   
   
  
# def convert_pdf_to_image(file, path):
    
#     pdf_file = fitz.open(stream=file.read(), filetype="pdf")
#     # pdf_file = fitz.open(file)  # Open the PDF
    
#     # Case 1: PDF has only one page
#     if pdf_file.page_count == 1:
#         page = pdf_file[0]  # Load the first page
#         image_list = page.get_images(full=True)
        
#         if len(image_list) > 0:  # Ensure the page has an image
#             xref = image_list[0][0]  # Get image reference
#             base_image = pdf_file.extract_image(xref)
            
#             image_bytes = base_image["image"]  # Extract raw image data
#             img = Image.open(io.BytesIO(image_bytes))  # Convert to a PIL image
            
#             img.save(path)  # Save the image
#             return True
#         else:
#             print("No images found in the PDF.")
#             return False

#     # Case 2: PDF has two pages, merge images
#     elif pdf_file.page_count == 2:
#         images = []  # List to store extracted images
        
#         for i in range(2):
#             page = pdf_file[i]  # Load each page
#             image_list = page.get_images(full=True)

#             if len(image_list) > 0:  # Ensure page has an image
#                 xref = image_list[0][0]
#                 base_image = pdf_file.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 img = Image.open(io.BytesIO(image_bytes))
#                 img = img.rotate(90, expand=True)

#                 images.append(img)
#             else:
#                 print(f"No images found on page {i+1}.")
#                 return False

#         # Merge the two images vertically
#         new_width = max(images[0].width, images[1].width)
#         new_height = images[0].height + images[1].height
#         merged_img = Image.new("RGB", (new_width, new_height), (255, 255, 255))

#         # Paste images onto the merged canvas
#         merged_img.paste(images[0], (0, 0))
#         merged_img.paste(images[1], (0, images[0].height))
        
#         merged_img.save(path)  # Save the merged image
#         return True
    
#     print("PDF has more than two pages. Function supports only 1 or 2 pages.")
#     return False      
    



#     return extracted_data
def extract_old_pan_data(text):

    extracted_data = {
        'ocr_name': '',
        'ocr_father_name': '',
        'ocr_dob': '',
        'ocr_pan_number': ''
    }
    
    print(text,"ye ocr text hai")

    # Remove unwanted characters and split text into lines
    lines = [re.sub(r'[^A-Za-z0-9 /,-]+', '', line).strip() for line in text.split('\n') if line.strip()]
    print("OCR Lines:", lines)
    

    for i, line in enumerate(lines):
        clean_line = re.sub(r'[^A-Za-z0-9 /-]+', '', line)

        # Detect PAN Number (Format: ABCDE1234F with error correction)
        pan_match = re.search(r'[A-Z]{5}[0-9A-Z]{4}[A-Z]', clean_line)
        if pan_match:
            correct_pan=correct_pan_number(pan_match.group(0))
            if len(correct_pan)==10:
                pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', correct_pan)
                if pan_match:
                
                    extracted_data['ocr_pan_number'] = correct_pan_number(pan_match.group(0))
                    continue

        # Detect Date of Birth (Format: DD/MM/YYYY or DD-MM-YYYY)
        dob_match = re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
        if dob_match:
            extracted_data['ocr_dob'] = normalize_dob_format(dob_match.group(0))
            continue

        # Detect Name (First uppercase text after "Name" keyword)
        if "TAX" in clean_line.upper() and i + 1 < len(lines) and  "INDIA" in re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper()      and not extracted_data.get('ocr_name') and  i + 2 < len(lines) :
            
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', lines[i + 2]).strip()
            continue
        elif "TAX" in clean_line.upper() and i + 1 < len(lines) and  "INDIA" not in re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper() and not extracted_data.get('ocr_name')  :
             extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', lines[i + 1]).strip()
            
            
        # Detect Father's Name (Occurs after "FATHER" keyword)
        if extracted_data.get('ocr_name')  and not extracted_data.get('ocr_dob')  :
            extracted_data['ocr_father_name'] = re.sub(r'[^A-Za-z ]+', '', lines[i]).strip()
            continue

    # Final cleanup: Remove extra characters from extracted fields
    extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', extracted_data['ocr_name']).strip()
    extracted_data['ocr_father_name'] = re.sub(r'[^A-Za-z ]+', '', extracted_data['ocr_father_name']).strip()
    extracted_data['ocr_pan_number'] = re.sub(r'[^A-Z0-9]+', '', extracted_data['ocr_pan_number']).strip()

    return extracted_data
    
    
def extract_pan_data(text):
    """
    Extracts PAN card details dynamically from OCR text lines, including Name, Father's Name, DOB, and PAN Number.
    """
    extracted_data = {
        'ocr_name': '',
        'ocr_father_name': '',
        'ocr_dob': '',
        'ocr_pan_number': ''
    }
    
    print(text,"ye ocr text hai")

    # Remove unwanted characters and split text into lines
    lines = [re.sub(r'[^A-Za-z0-9 /,-]+', '', line).strip() for line in text.split('\n') if line.strip()]
    print("OCR Lines:", lines)
    

    for i, line in enumerate(lines):
        clean_line = re.sub(r'[^A-Za-z0-9 /-]+', '', line)

        # Detect PAN Number (Format: ABCDE1234F with error correction)
        pan_match = re.search(r'[A-Z]{5}[0-9A-Z]{4}[A-Z]', clean_line)
        if pan_match:
            correct_pan=correct_pan_number(pan_match.group(0))
            if len(correct_pan)==10:
                pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]', correct_pan)
                if pan_match:
                
                    extracted_data['ocr_pan_number'] = correct_pan_number(pan_match.group(0))
                    continue

        # Detect Date of Birth (Format: DD/MM/YYYY or DD-MM-YYYY)
        dob_match = re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
        if dob_match and not extracted_data['ocr_dob']:
            extracted_data['ocr_dob'] = normalize_dob_format(dob_match.group(0))
            continue

        # Detect Name (First uppercase text after "Name" keyword)
        if "NAME" in clean_line.upper() and i + 1 < len(lines) and not extracted_data['ocr_name'] and not any(keyword in line for keyword in ["GOVT.", "INCOME TAX", "PERMANENT ACCOUNT NUMBER", "CARD", "DATE OF BIRTH", "FATHER"]):
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', lines[i + 1]).strip()
            continue

        # Detect Father's Name (Occurs after "FATHER" keyword)
        if "FATHER" in clean_line.upper() and i + 1 < len(lines):
            extracted_data['ocr_father_name'] = re.sub(r'[^A-Za-z ]+', '', lines[i + 1]).strip()
            continue

    # Final cleanup: Remove extra characters from extracted fields
    extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', extracted_data['ocr_name']).strip()
    extracted_data['ocr_father_name'] = re.sub(r'[^A-Za-z ]+', '', extracted_data['ocr_father_name']).strip()
    extracted_data['ocr_pan_number'] = re.sub(r'[^A-Z0-9]+', '', extracted_data['ocr_pan_number']).strip()

    return extracted_data
# def preprocess_image(image):
#     """
#     Preprocesses the image for better OCR results on Indian PAN cards.
#     """
#     img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    
#     img = cv2.GaussianBlur(img, (3, 3), 0)
    
#     _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return Image.fromarray(img)



def DigiLockeraadhar(text):
    extracted_data = {
        'ocr_name': '',
        'ocr_address': '',
        'ocr_dob': '',
        'ocr_adhar_number': '',
        "ocr_address_for_match":""
    }
    lines = [re.sub(r'[^A-Za-z0-9 /,-]+', '', line).strip() for line in text.split('\n') if line.strip()]
    print("OCR Lines:", lines)
    if "DIGILOCKER" in lines[0].upper():
        
        
        for i, line in enumerate(lines):
            clean_line = re.sub(r'[^A-Za-z0-9 /-]+', '', line)
            
            
        
            if "masked aadhaar number" in    clean_line.lower():
                if not  extracted_data['ocr_adhar_number']:
                
                    extracted_data['ocr_adhar_number']=re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).strip() 
                    continue
                
            if "name" == clean_line.lower().strip() :
                if not  extracted_data['ocr_name']  :
                    ocr_name=re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).strip()
                    if "date of birth"  in ocr_name.lower() or "photo" in ocr_name.lower() :
                        for j in range(1,i):
                            ocr_name=re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i-j]).strip().lower()
                            if "date of birth" not  in ocr_name and  "photo" not  in ocr_name :
                                extracted_data['ocr_name']=ocr_name
                                break
                            
                            if "masked aadhaar number" in    ocr_name:
                                break
                    else:
                        extracted_data['ocr_name']=ocr_name               
                    continue
             
            if "date of birth" == clean_line.lower().strip():
                
                if not  extracted_data['ocr_dob']:
                    
                    dob=re.sub(r'[^0-9 /-]+', '', lines[i+1]).strip() 
                    if dob:
                        # extracted_data['ocr_dob']=re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).strip() 
                        extracted_data['ocr_dob']=dob 
                        
                    elif re.sub(r'[^0-9 /-]+', '', lines[i-1]).strip() :
                        
                        extracted_data['ocr_dob']=re.sub(r'[^0-9 /-]+', '', lines[i-1]).strip() 
                    elif re.sub(r'[^0-9 /-]+', '', lines[i+2]).strip() :
                        extracted_data['ocr_dob']=re.sub(r'[^0-9 /-]+', '', lines[i+2]).strip()
                
            if  "c/o" in  clean_line.lower().strip() and not extracted_data['ocr_address'] :
                
                
                
            
                for j in range(i+1,len(lines)):
                    
                    clean_line1 = re.sub(r'[^A-Za-z0-9 /-]+', '', lines[j])
                    
                   
                    if "CONFIDENTAL" in clean_line1.upper():
                        break
                        
                        
                        
                    if  not extracted_data['ocr_address']: 
                        extracted_data['ocr_address']=extracted_data['ocr_address']+clean_line1
                    
                    extracted_data['ocr_address_for_match']=extracted_data['ocr_address_for_match']+clean_line1    
                    extracted_data['ocr_address']=extracted_data['ocr_address']+","+clean_line1
                
                
                
                
            elif "s/o" in  clean_line.lower().strip() and not extracted_data['ocr_address']:
                for j in range(i+1,len(lines)):
                    
                    clean_line1 = re.sub(r'[^A-Za-z0-9 /-]+', '', lines[j])
                    
                    
                    if "CONFIDENTAL" in clean_line1.upper():
                        break
                        
                        
                        
                    if  not extracted_data['ocr_address']: 
                        extracted_data['ocr_address']=extracted_data['ocr_address']+clean_line1
                    
                    extracted_data['ocr_address_for_match']=extracted_data['ocr_address_for_match']+clean_line1    
                    extracted_data['ocr_address']=extracted_data['ocr_address']+","+clean_line1
                
            elif "d/o" in  clean_line.lower().strip() and not extracted_data['ocr_address']:
                for j in range(i+1,len(lines)):
                    
                    clean_line1 = re.sub(r'[^A-Za-z0-9 /-]+', '', lines[j])
                    
                    
                    if "CONFIDENTAL" in clean_line1.upper():
                        break
                        
                        
                        
                    if  not extracted_data['ocr_address']: 
                        extracted_data['ocr_address']=extracted_data['ocr_address']+clean_line1
                    
                    extracted_data['ocr_address_for_match']=extracted_data['ocr_address_for_match']+clean_line1    
                    extracted_data['ocr_address']=extracted_data['ocr_address']+","+clean_line1
            
            
            
            
            
            
            
            
            
        return extracted_data
    
    else:
        return extracted_data
        
    
    
def extract_front_page_reissue_adahar(text):
    extracted_data = {
        'ocr_name': '',
        'ocr_address': '',
        'ocr_dob': '',
        'ocr_adhar_number': '',
        "ocr_address_for_match":""
    }
     
    lines = [re.sub(r'[^A-Za-z0-9 /,-]+', '', line).strip() for line in text.split('\n') if line.strip()]
    print("OCR Lines:", lines)
    for i, line in enumerate(lines):
        clean_line = re.sub(r'[^A-Za-z0-9 /-]+', '', line)
        
        adharno= re.sub(r'[^0-9]+', '', clean_line).strip()
        if len(adharno)==12:
            extracted_data['ocr_adhar_number']=adharno 
            continue
            
            
        
        if "DOB"  in clean_line.upper() and i + 1 < len(lines)  and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                    
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
                    
                    
            except:
                pass
            
            
        elif "D08"  in clean_line.upper() and i + 1 < len(lines)  and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                    
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
                    
                    
            except:
                pass
        elif "DB"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            print(ocr_dob,"ocr_dob")
            try:
                if ocr_dob[0]=="/":
                   ocr_dob= ocr_dob[1:]
                    
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                print(ocr_dob,"ocr_dob1")
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
            
        elif "D8"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            print(ocr_dob,"ocr_dob")
            try:
                if ocr_dob[0]=="/":
                   ocr_dob= ocr_dob[1:]
                    
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                print(ocr_dob,"ocr_dob1")
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
                
        elif "DO"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
        elif "D0"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
        # if "GOVERNMENT" in clean_line.strip().upper() and i + 2 < len(lines) and not extracted_data['ocr_name'] and not extracted_data['ocr_dob']    and  len(re.sub(r'[^0-9 ]+', '', lines[i + 2]).strip())==8 :
        #     extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', lines[i + 1]).strip()
        #     continue    
        # elif "INDIA" in clean_line.strip().upper()  and i + 2 < len(lines)  and not extracted_data['ocr_name']  and extracted_data['ocr_dob'] and  len(re.sub(r'[^0-9 ]+', '', lines[i + 2]).strip())==8:
        #     extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', lines[i + 1]).strip()
        #     continue  
        
        namecheck=""    
        if i+1< len(lines):
            namecheck=re.sub(r'[^0-9]+', '', lines[i + 1]).strip()
            
            if len(namecheck)>8:
                if namecheck[0]=="0":
                    namecheck=namecheck[1:]
                if namecheck[0]=="8":
                    namecheck=namecheck[1:]
                    
            
        if len(namecheck)==8 and not extracted_data['ocr_name'] and not extracted_data['ocr_dob'] :
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', clean_line).strip()
            continue 
    print("extract_front_page_reissue_adahar")
        
    return extracted_data

def extract_front_page_adahar(text):
    
    extracted_data = {
        'ocr_name': '',
        'ocr_address': '',
        'ocr_dob': '',
        'ocr_adhar_number': '',
        "ocr_address_for_match":""
    }
    
    lines = [re.sub(r'[^A-Za-z0-9 /,-]+', '', line).strip() for line in text.split('\n') if line.strip()]
    print("OCR Lines:", lines)
    for i, line in enumerate(lines):
        clean_line = re.sub(r'[^A-Za-z0-9 /-]+', '', line)
        
        adharno= re.sub(r'[^0-9]+', '', clean_line).strip()
        if len(adharno)==12:
            extracted_data['ocr_adhar_number']=adharno 
            continue
            
            
        
        if "DOB"  in clean_line.upper() and i + 1 < len(lines)  and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                    
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
                    
                    
            except:
                pass
        elif "D08"  in clean_line.upper() and i + 1 < len(lines)  and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                    
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
                    
                    
            except:
                pass
        elif "DB"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            print(ocr_dob,"ocr_dob")
            try:
                if ocr_dob[0]=="/":
                   ocr_dob= ocr_dob[1:]
                    
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                print(ocr_dob,"ocr_dob1")
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
            
        elif "D8"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            print(ocr_dob,"ocr_dob")
            try:
                if ocr_dob[0]=="/":
                   ocr_dob= ocr_dob[1:]
                    
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                print(ocr_dob,"ocr_dob1")
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
                
        elif "DO"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
        
        
        elif "D0"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
            
            
        # if "DATE" in clean_line.strip().upper() and not extracted_data['ocr_name'] and not extracted_data['ocr_dob'] :
        #     extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', lines[i + 1]).strip()
        #     continue    
        # elif "ISSUE" in clean_line.strip().upper() and not extracted_data['ocr_name']  and extracted_data['ocr_dob']:
        #     extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', lines[i + 1]).strip()
        #     continue  
        
        
        namecheck=""    
        if i+1< len(lines):
            namecheck=re.sub(r'[^0-9]+', '', lines[i + 1]).strip()
            
            if len(namecheck)>8:
                if namecheck[0]=="0":
                    namecheck=namecheck[1:]
                if namecheck[0]=="8":
                    namecheck=namecheck[1:]
        
        
        
        if len(namecheck)==8       and "DO" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip()  and      not extracted_data['ocr_name'] and not extracted_data['ocr_dob'] :
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', clean_line).strip()
            continue   
        elif    len(namecheck)==8       and "D0" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip()  and      not extracted_data['ocr_name'] and not extracted_data['ocr_dob']:
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', clean_line).strip()
            continue  
            
        elif    len(namecheck)==8       and "DOB" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip()  and      not extracted_data['ocr_name'] and not extracted_data['ocr_dob']:
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', clean_line).strip()
            continue
        elif    len(namecheck)==8       and "D08" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip()  and      not extracted_data['ocr_name'] and not extracted_data['ocr_dob']:
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', clean_line).strip()
            continue
        
        elif    len(namecheck)==8       and "DB" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip()  and      not extracted_data['ocr_name'] and not extracted_data['ocr_dob']:
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', clean_line).strip()
            continue
        elif    len(namecheck)==8       and "D8" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip()  and      not extracted_data['ocr_name'] and not extracted_data['ocr_dob']:
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', clean_line).strip()
            continue
    print("extract_front_page_adahar")
           
    return extracted_data     
        
    

def extract_full_page_adahar(text):
    extracted_data = {
        'ocr_name': '',
        'ocr_address': '',
        'ocr_dob': '',
        'ocr_adhar_number': '',
        "ocr_address_for_match":""
    }
    flag=True
    lines = [re.sub(r'[^A-Za-z0-9 /,-]+', '', line).strip() for line in text.split('\n') if line.strip()]
    print("OCR Lines:", lines)
    for i, line in enumerate(lines):
        clean_line = re.sub(r'[^A-Za-z0-9 /-]+', '', line)
        
        if "AADHAAR" in clean_line.upper() and i + 1 < len(lines):
            adharno= re.sub(r'[^0-9]+', '', lines[i + 1]).strip()
           
            if len(adharno)==12:
                extracted_data['ocr_adhar_number']=adharno 
                continue

        if "DOB"  in clean_line.upper() and i + 1 < len(lines)  and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                    
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
                    
                    
            except:
                pass
            
        elif "D08"  in clean_line.upper() and i + 1 < len(lines)  and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                    
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
                    
                    
            except:
                pass
        elif "DB"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            print(ocr_dob,"ocr_dob")
            try:
                if ocr_dob[0]=="/":
                   ocr_dob= ocr_dob[1:]
                    
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                print(ocr_dob,"ocr_dob1")
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
            
        elif "D8"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            print(ocr_dob,"ocr_dob")
            try:
                if ocr_dob[0]=="/":
                   ocr_dob= ocr_dob[1:]
                    
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                print(ocr_dob,"ocr_dob1")
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
                
        elif "DO"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
            
        elif "D0"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
            
        if clean_line.strip().upper()=="TO":
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', lines[i + 1]).strip()
            flag=False
            continue
        
        
        
        if extracted_data['ocr_name'] and not extracted_data['ocr_adhar_number'] and not extracted_data['ocr_address'] and   flag :
            
            
            for j in range(i,len(lines)):
                
                clean_line1 = re.sub(r'[^A-Za-z0-9 /-]+', '', lines[j])
                
                ocr_pin=re.sub(r'[^0-9 ]+', '', clean_line1).strip()
                if "PIN" in clean_line1 and len(ocr_pin)==6:
                     extracted_data['ocr_address']=extracted_data['ocr_address']+","+clean_line1
                     extracted_data['ocr_address_for_match']=extracted_data['ocr_address_for_match']+clean_line1
                     
                     break
                elif len(ocr_pin)==6:
                     extracted_data['ocr_address']=extracted_data['ocr_address']+","+clean_line1
                     extracted_data['ocr_address_for_match']=extracted_data['ocr_address_for_match']+clean_line1
                     break
                    
                    
                if  not extracted_data['ocr_address']: 
                    extracted_data['ocr_address']=extracted_data['ocr_address']+clean_line1
                
                extracted_data['ocr_address_for_match']=extracted_data['ocr_address_for_match']+clean_line1    
                extracted_data['ocr_address']=extracted_data['ocr_address']+","+clean_line1
                
        flag=True      
            
            
            
        
            
            
    print("extract_full_page_adahar")
         
    return extracted_data    
            
            
        
def extract_both_side_adahar(text) :
    print("extract_both_side_adahar")
    
    extracted_data = {
        'ocr_name': '',
        'ocr_address': '',
        'ocr_dob': '',
        'ocr_adhar_number': '',
        "ocr_address_for_match":""
    }
    
    flag= False
    
    lines = [re.sub(r'[^A-Za-z0-9 /,-]+', '', line).strip() for line in text.split('\n') if line.strip()]
    print("OCR Lines:", lines)
    for i, line in enumerate(lines):
        clean_line = re.sub(r'[^A-Za-z0-9 /-]+', '', line)
        
        adharno= re.sub(r'[^0-9]+', '', clean_line).strip()
        if len(adharno)==12:
            extracted_data['ocr_adhar_number']=adharno 
            continue
            
            
        
        if "DOB"  in clean_line.upper() and i + 1 < len(lines)  and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                    
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
                    
                    
            except:
                pass
        elif "D08"  in clean_line.upper() and i + 1 < len(lines)  and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                    
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
                    
                    
            except:
                pass
        elif "DB"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            print(ocr_dob,"ocr_dob")
            try:
                if ocr_dob[0]=="/":
                   ocr_dob= ocr_dob[1:]
                    
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                print(ocr_dob,"ocr_dob1")
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
            
        elif "D8"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            print(ocr_dob,"ocr_dob")
            try:
                if ocr_dob[0]=="/":
                   ocr_dob= ocr_dob[1:]
                    
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                print(ocr_dob,"ocr_dob1")
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
                
        elif "DO"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
        
        
        elif "D0"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
            
            
        # if "DATE" in clean_line.strip().upper() and not extracted_data['ocr_name'] and not extracted_data['ocr_dob'] :
        #     extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', lines[i + 1]).strip()
        #     continue    
        # elif "ISSUE" in clean_line.strip().upper() and not extracted_data['ocr_name']  and extracted_data['ocr_dob']:
        #     extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', lines[i + 1]).strip()
        #     continue  
            
        namecheck=""    
        if i+1< len(lines):
            namecheck=re.sub(r'[^0-9]+', '', lines[i + 1]).strip()
            
            if len(namecheck)>8:
                if namecheck[0]=="0":
                    namecheck=namecheck[1:]
                if namecheck[0]=="8":
                    namecheck=namecheck[1:]
        
        
        
        if len(namecheck)==8       and "DO" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip()  and      not extracted_data['ocr_name'] and not extracted_data['ocr_dob'] :
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', clean_line).strip()
            continue   
        elif    len(namecheck)==8       and "D0" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip()  and      not extracted_data['ocr_name'] and not extracted_data['ocr_dob']:
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', clean_line).strip()
            continue  
            
        elif    len(namecheck)==8       and "DOB" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip()  and      not extracted_data['ocr_name'] and not extracted_data['ocr_dob']:
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', clean_line).strip()
            continue
        elif    len(namecheck)==8       and "D08" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip()  and      not extracted_data['ocr_name'] and not extracted_data['ocr_dob']:
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', clean_line).strip()
            continue
        
        elif    len(namecheck)==8       and "DB" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip()  and      not extracted_data['ocr_name'] and not extracted_data['ocr_dob']:
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', clean_line).strip()
            continue
        elif    len(namecheck)==8       and "D8" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip()  and      not extracted_data['ocr_name'] and not extracted_data['ocr_dob']:
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', clean_line).strip()
            continue    
            
           
            
        if extracted_data['ocr_name'] and extracted_data['ocr_adhar_number'] and not extracted_data['ocr_address'] and   flag :
            
            
            for j in range(i,len(lines)):
                
                clean_line1 = re.sub(r'[^A-Za-z0-9 /-]+', '', lines[j])
                
                ocr_pin=re.sub(r'[^0-9 ]+', '', clean_line1).strip()
                if "PIN" in clean_line1 and len(ocr_pin)==6:
                     extracted_data['ocr_address']=extracted_data['ocr_address']+","+clean_line1
                     extracted_data['ocr_address_for_match']=extracted_data['ocr_address_for_match']+clean_line1
                     
                     break
                elif len(ocr_pin)==6:
                     extracted_data['ocr_address']=extracted_data['ocr_address']+","+clean_line1
                     extracted_data['ocr_address_for_match']=extracted_data['ocr_address_for_match']+clean_line1
                     break
                    
                    
                if  not extracted_data['ocr_address']: 
                    extracted_data['ocr_address']=extracted_data['ocr_address']+clean_line1
                
                extracted_data['ocr_address_for_match']=extracted_data['ocr_address_for_match']+clean_line1    
                extracted_data['ocr_address']=extracted_data['ocr_address']+","+clean_line1
                
        if "ADDRESS" in clean_line.upper():
            flag=True
            
        
        
        
        
        
        
    return extracted_data  
    
    
    
def extract_both_side_reissue_adahar(text) :
    
    
        
    extracted_data = {
        'ocr_name': '',
        'ocr_address': '',
        'ocr_dob': '',
        'ocr_adhar_number': '',
        "ocr_address_for_match":""
    }
    
    flag= False
    
    lines = [re.sub(r'[^A-Za-z0-9 /,-]+', '', line).strip() for line in text.split('\n') if line.strip()]
    print("OCR Lines:", lines)
    for i, line in enumerate(lines):
        clean_line = re.sub(r'[^A-Za-z0-9 /-]+', '', line)
        
        adharno= re.sub(r'[^0-9]+', '', clean_line).strip()
        if len(adharno)==12:
            extracted_data['ocr_adhar_number']=adharno 
            continue
            
            
        
        if "DOB"  in clean_line.upper() and i + 1 < len(lines)  and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                    
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
                    
                    
            except:
                pass
        elif "D08"  in clean_line.upper() and i + 1 < len(lines)  and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                    
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
                    
                    
            except:
                pass
        elif "DB"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            print(ocr_dob,"ocr_dob")
            try:
                if ocr_dob[0]=="/":
                   ocr_dob= ocr_dob[1:]
                    
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                print(ocr_dob,"ocr_dob1")
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
            
        elif "D8"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            print(ocr_dob,"ocr_dob")
            try:
                if ocr_dob[0]=="/":
                   ocr_dob= ocr_dob[1:]
                    
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                print(ocr_dob,"ocr_dob1")
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
                
        elif "DO"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
        
        
        elif "D0"  in clean_line.upper() and i + 1 < len(lines) and "MALE" in  re.sub(r'[^A-Za-z0-9 /-]+', '', lines[i+1]).upper().strip() :
            ocr_dob=  re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', clean_line)
            if ocr_dob:
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob.group(0))
                continue
            ocr_dob=re.sub(r'[^0-9 /-]+', '', clean_line).strip()
            try:
                if ocr_dob[0]=="/":
                       ocr_dob= ocr_dob[1:]
                if ocr_dob[2]!='/':
                    ocr_dob=ocr_dob[:2]+"/"+ocr_dob[2:]
                if  ocr_dob[5]!='/':
                    ocr_dob=ocr_dob[:5]+"/"+ocr_dob[5:]
                extracted_data['ocr_dob'] = normalize_dob_format(ocr_dob)
                continue
                    
            except:
                pass
            
   
            
            
        namecheck=""    
        if i+1< len(lines):
            namecheck=re.sub(r'[^0-9]+', '', lines[i + 1]).strip()
            
            if len(namecheck)>8:
                if namecheck[0]=="0":
                    namecheck=namecheck[1:]
                if namecheck[0]=="8":
                    namecheck=namecheck[1:]
                    
                    
            
            
        if len(namecheck)==8 and not extracted_data['ocr_name'] and not extracted_data['ocr_dob'] :
            extracted_data['ocr_name'] = re.sub(r'[^A-Za-z ]+', '', clean_line).strip()
            continue 
           
            
        if extracted_data['ocr_name'] and extracted_data['ocr_adhar_number'] and not extracted_data['ocr_address'] and   flag :
            
            
            for j in range(i,len(lines)):
                
                clean_line1 = re.sub(r'[^A-Za-z0-9 /-]+', '', lines[j])
                
                ocr_pin=re.sub(r'[^0-9 ]+', '', clean_line1).strip()
                if "PIN" in clean_line1 and len(ocr_pin)==6:
                     extracted_data['ocr_address']=extracted_data['ocr_address']+","+clean_line1
                     extracted_data['ocr_address_for_match']=extracted_data['ocr_address_for_match']+clean_line1
                     
                     break
                elif len(ocr_pin)==6:
                     extracted_data['ocr_address']=extracted_data['ocr_address']+","+clean_line1
                     extracted_data['ocr_address_for_match']=extracted_data['ocr_address_for_match']+clean_line1
                     break
                    
                    
                if  not extracted_data['ocr_address']: 
                    extracted_data['ocr_address']=extracted_data['ocr_address']+clean_line1
                
                extracted_data['ocr_address_for_match']=extracted_data['ocr_address_for_match']+clean_line1    
                extracted_data['ocr_address']=extracted_data['ocr_address']+","+clean_line1
                
        if "ADDRESS" in clean_line.upper():
            flag=True
            
        
        
    print("extract_both_side_reissue_adahar")
    
        
        
        
    return extracted_data 
      
        
        
        
    
    
    



def preprocess_image(image):
    """
    Preprocesses the image for better OCR results on Indian PAN cards.
    """
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  # Directly convert to grayscale
    
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)  # Better upscale

    img = cv2.GaussianBlur(img, (3, 3), 0)  # Reduce noise
    
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding
    
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)  # Slightly bolden text

    return Image.fromarray(img)

def extract_bank_data(text):
    """
    Extracts IFSC, MICR, and Account Holder Name from OCR text.
    Handles cheques, bank statements, and other bank-related documents.
    """
    info = {
        "ocr_IFSC": None,
        "ocr_MICR": None,
        "ocr_Account_Holder_Name": None
    }

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)

    ifsc = re.search(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', text)
    if ifsc:
        info["ocr_IFSC"] = ifsc.group(0)

    if not info["ocr_IFSC"]:
        corrected_text = text.replace('8', 'B').replace('O', '0')  
        ifsc_corrected = re.search(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', corrected_text)
        if ifsc_corrected:
            info["ocr_IFSC"] = ifsc_corrected.group(0)

    micr = re.search(r'MICR\s*Code\s*:\s*(\d{9})', text, re.IGNORECASE)
    if micr:
        info["ocr_MICR"] = micr.group(1)

    if not info["ocr_MICR"]:
        micr_fallback = re.search(r'\b\d{9}\b', text)
        if micr_fallback:
            info["ocr_MICR"] = micr_fallback.group(0)

    for i, line in enumerate(lines):
        if "Please sign above" in line or "Payable at" in line:
            for j in range(1, 4):  
                if i - j >= 0:
                    potential_name = lines[i - j].strip()
                    if len(potential_name.split()) > 1 and not re.search(r'(Savings|Account|Branch|Code|Bearer)', potential_name, re.IGNORECASE):
                        validated_name = re.sub(r'[^A-Za-z\s&]+', '', potential_name).strip()
                        if validated_name and not re.search(r'\bon\b', validated_name, re.IGNORECASE):
                            info["ocr_Account_Holder_Name"] = validated_name
                            break
            break

    if not info["ocr_Account_Holder_Name"]:
        name_context_pattern = r'1©SC CODE.*?\}\s*([A-Z][A-Za-z\s]+)'
        name_context_match = re.search(name_context_pattern, text, re.IGNORECASE)
        if name_context_match:
            info["ocr_Account_Holder_Name"] = name_context_match.group(1).strip()

    if not info["ocr_Account_Holder_Name"]:
        name_pattern_statement = r'(?:MR\.|MRS\.|ACCOUNT HOLDER|NAME\(S\))\s*([A-Z][A-Za-z\s&]+(?:\s[A-Z][A-Za-z\s&]+)?)'
        name_statement = re.search(name_pattern_statement, text, re.IGNORECASE)
        if name_statement:
            info["ocr_Account_Holder_Name"] = name_statement.group(1).strip()

    if not info["ocr_Account_Holder_Name"]:
        name_fallback = re.search(r'\b[A-Z][a-z]+\s[A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)?\b', text)
        if name_fallback:
            potential_name = name_fallback.group(0).strip()
            if not re.search(r'(Bank|Statement|Account|Branch|Bearer)', potential_name, re.IGNORECASE):
                info["ocr_Account_Holder_Name"] = potential_name

    if info["ocr_Account_Holder_Name"]:
        info["ocr_Account_Holder_Name"] = re.sub(r'\s{2,}', ' ', info["ocr_Account_Holder_Name"]).strip()

    
    account = re.search(r'Account\s*Number\s*:\s*(\d{8,18})', text, re.IGNORECASE)
    if account:
        info["ocr_account_number"] = account.group(1)

    # Fallback: If the specific pattern wasn't found, search for any sequence of 8 to 18 digits in the text.
    if not info.get("ocr_account_number"):
        account_fallback = re.search(r'\b\d{8,18}\b', text)
        if account_fallback:
            info["ocr_account_number"] = account_fallback.group(0)
    
    return info

def savelogs(log_entry,log_file_path):
    if os.path.exists(log_file_path):
        with open(log_file_path, "r+", encoding="utf-8") as file:
            try:
                logs = json.load(file)  # Load existing logs
                if isinstance(logs, list):
                    logs.append(log_entry)
                else:
                    logs = [logs, log_entry]
            except json.JSONDecodeError:
                logs = [log_entry]  # If file is empty or corrupted

            file.seek(0)
            json.dump(logs, file, indent=4)
    else:
        with open(log_file_path, "w", encoding="utf-8") as file:
            json.dump([log_entry], file, indent=4)

def get_index(text,PAN,first_NAME,middle_NAME,last_NAME,FATHER_first_NAME,FATHER_middle_NAME,FATHER_last_NAME,dob):
    
    # text = """farrsr HRCR INCOME TAX DEPARTMENT GOVT.UFINDIA  lhohehlk PermanentAccount Number Card AQGPA7252E /Name ARECONTY KARLNAKAR PRASAD fot  a /Father's Name YELLAPPA ARECONTY Anha  t art /Date of Birth /Signature5630612 08/07/1971"""

    # PAN = "AQGPA7252E"
    # first_NAME = "ARECONTY"
    # middle_NAME = "KARLNAKAR"
    # last_NAME = "PRASAD"
    # FATHER_first_NAME = "YELLAPPA"
    # FATHER_middle_NAME = ""
    # FATHER_last_NAME = "ARECONTY"
    # dob = "08/07/1971"

    def find_indices(text, word):
        """Find all occurrences of a word in text and return their start and end indexes"""
        matches = [match.span() for match in re.finditer(re.escape(word), text)]
        return matches if matches else None

    # Store extracted positions
    positions = {}
    result_list = []

    for label, word in [
        ("PAN", PAN),
        ("first_NAME", first_NAME),
        ("middle_NAME", middle_NAME),
        ("last_NAME", last_NAME),
        ("FATHER_first_NAME", FATHER_first_NAME),
        ("FATHER_middle_NAME", FATHER_middle_NAME),
        ("FATHER_last_NAME", FATHER_last_NAME),
        ("DOB", dob)
    ]:
        if word:
            indices = find_indices(text, word)
            if indices:
                positions[label] = indices  # Store all occurrences
                for start, end in indices:
                    result_list.append((start, end, label))

    return str(result_list)


@app.route('/get_pan_text', methods=['POST'])
def get_pan_text():
    
    file = request.files['panfile']
    PAN = request.form.get('PAN')
    first_NAME = request.form.get('first_NAME')
    middle_NAME = request.form.get('middle_NAME')
    last_NAME = request.form.get('last_NAME')
    
    FATHER_first_NAME = request.form.get('FATHER_first_NAME')
    FATHER_middle_NAME = request.form.get('FATHER_middle_NAME')
    FATHER_last_NAME = request.form.get('FATHER_last_NAME')
    dob = request.form.get('dob')
    path=f"static/training/pan/"
    if not os.path.exists(path):
            os.makedirs(path)
    
    extension = os.path.splitext(file.filename)[1]
    new_filename = f"{PAN}{extension}"
    
        # Full path for saving the file
    save_path = os.path.join(path, new_filename)
    
        # save_path = os.path.join(path, new_filename)
    file.save(save_path)
    
    
    
    
    
    
    ocr = PaddleOCR(use_angle_cls=True, lang="en" ,use_mp=False,  cpu_threads=4,det_model_dir='models/ch_PP-OCRv3_det_infer', rec_model_dir='models/ch_PP-OCRv3_rec_infer')
    results = ocr.ocr(save_path, cls=True)

        # Convert OCR results into a single text block
        # text = "\n".join([line[1][0] for result in results for line in result])
    text = " ".join([line[1][0] for result in results for line in result])
    
    indiexdata=get_index(text,PAN,first_NAME,middle_NAME,last_NAME,FATHER_first_NAME,FATHER_middle_NAME,FATHER_last_NAME,dob)
    
    print(indiexdata)
    
    return {"text":text,"index_data":indiexdata}
    
    
    
    

@app.route('/pan_training', methods=['POST'])
def pan_training():
    # file = request.files['panfile']
    # first_NAME = request.form.get('first_NAME')
    # middle_NAME = request.form.get('middle_NAME')
    # last_NAME = request.form.get('last_NAME')
    
    # FATHER_first_NAME = request.form.get('FATHER_first_NAME')
    # FATHER_middle_NAME = request.form.get('FATHER_middle_NAME')
    # FATHER_last_NAME = request.form.get('FATHER_last_NAME')
    # PAN = request.form.get('PAN')
    # dob = request.form.get('dob')
    text = request.form.get('text')
    data=text = request.form.get('data')
    data=data=eval(data)
    
    
    # if PAN:
    #     pan_start = text.find(PAN)
    #     pan_end = pan_start + len(PAN)
    # if first_NAME:
    #     first_NAME_start = text.find(first_NAME)
    #     first_NAME_end = first_NAME_start + len(first_NAME)
    # if middle_NAME:    
    #     middle_NAME_start = text.find(middle_NAME)
    #     middle_NAME_end = middle_NAME_start + len(middle_NAME)
    # if last_NAME:    
    #     last_NAME_start = text.find(last_NAME)
    #     last_NAME_end = last_NAME_start + len(last_NAME)
    # if FATHER_first_NAME:    
    #     FATHER_first_NAME_start = text.find(FATHER_first_NAME)
    #     FATHER_first_NAME_end = FATHER_first_NAME_start + len(FATHER_first_NAME)
        
        
    # if FATHER_middle_NAME:    
    #     FATHER_middle_NAME_start = text.find(FATHER_middle_NAME)
    #     FATHER_middle_NAME_end = FATHER_middle_NAME_start + len(FATHER_middle_NAME)
    # if FATHER_last_NAME:    
    #     FATHER_last_NAME_start = text.find(FATHER_last_NAME)
    #     FATHER_last_NAME_end = FATHER_last_NAME_start + len(FATHER_last_NAME)
    # if dob:
    #     dob_start = text.find(dob)
    #     dob_end = dob_start + len(dob)
    
    
    # data=[]
    
    # if first_NAME and first_NAME in text :
    #     data.append((first_NAME_start,first_NAME_end,"first_NAME"))
        
    # if middle_NAME and middle_NAME in text :
    #     data.append((middle_NAME_start,middle_NAME_end,"middle_NAME"))
        
    # if last_NAME and  last_NAME in text :
    #     data.append((last_NAME_start,last_NAME_end,"last_NAME"))
        
    # if FATHER_first_NAME and  FATHER_first_NAME in text:
    #     data.append((FATHER_first_NAME_start,FATHER_first_NAME_end,"FATHER_first_NAME"))
        
    # if FATHER_middle_NAME and  FATHER_middle_NAME in text:
    #     data.append((FATHER_middle_NAME_start,FATHER_middle_NAME_end,"FATHER_middle_NAME"))
        
    # if FATHER_last_NAME and  FATHER_last_NAME in text:
    #     data.append((FATHER_last_NAME_start,FATHER_last_NAME_end,"FATHER_last_NAME"))
        
    
    # if dob and  dob in text:
    #     data.append((dob_start,dob_end,"DOB"))
        
    # if PAN and  PAN in text:
    #     data.append((pan_start,pan_end,"PAN"))
        
    
    
    TRAIN_DATA = [
    (text, 
    {"entities": data})]
    
    #msg= retrain_spacy_ner(TRAIN_DATA,model_path="/home/meon/ocr/custom_pan_ner",iterations=50)
    
    return {"msg":msg}
    
    
    
    
    
    

@app.route('/extract_pan_details', methods=['POST'])
@jwt_required() 
def extract_pan_details():
    print("Files received:", list(request.files.keys()))
    
    
    file = request.files['panfile']
    name = request.form.get('name')
    pan = request.form.get('pan')
    sources = request.form.get('sources')
    father_name = request.form.get('fathername')
    dob = request.form.get('dob')
    req_id = request.form.get('req_id')
    company = request.form.get('company')
    dob=convert_to_ddmmyyyy(dob)
    token=get_jwt_identity()

    sta,msg=check_auth(token,"pan_access")
    if not sta:
        return jsonify({'msg':msg ,"success":False})
    try:
        
    #     if 'clientcode' not in request.json:
    #         return jsonify({'msg': 'clientcode is required'}), 400

    #     clientcode = request.json['clientcode']
        
        
        
        
        
        request_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "headers": dict(request.headers),
        
        "files": list(request.files.keys()) ,
        "req_id":req_id,
        "sources" : sources
        
        
            }
        
        logger.info(f"request_data_extract_pan_details ",extra=request_data)
        
        
        
        if not sources:
            return {"msg":"Please provide sources","success":False}
        
        path=f"static/panfiles/{sources}"
        
        pathlog=f"static/logs/panfiles/{sources}"
        if not os.path.exists(pathlog):
            os.makedirs(pathlog)
            
        log_file_path = os.path.join(pathlog, f"{pan}.json")
            
        if not os.path.exists(path):
            os.makedirs(path)
            
        
        # new_filename = f"{pan}.jpg"  # You can adjust the file extension as needed
        extension = os.path.splitext(file.filename)[1]
        
        new_filename = f"{pan}{extension}"
        new_filename1=f"{pan}.jpg"
        # Full path for saving the file
        save_path = os.path.join(path, new_filename)
        save_path1 = os.path.join(path, new_filename1)
        # save_path = os.path.join(path, new_filename)
        if extension.lower()==".pdf":
            convert_pdf_to_image(file,save_path1)
            save_path=save_path1
            
        else:   
            file.save(save_path)
        
        
        
       
  


        
        
        # img = Image.open(save_path)
        # preprocessed_img = preprocess_image(img)
        # preprocessed_img = np.array(preprocessed_img)
        # print(preprocessed_img,type(preprocessed_img),"nknkj")
        
 
        # reader = easyocr.Reader(['en'], gpu=True) 
        # text= reader.readtext(img, detail=0)
        # text="\n".join(text) 
        # img = Image.open(save_path).convert("L")
        
        # custom_config = r'--oem 1 --psm 6 -c textord_heavy_nr=1 tessedit_write_images=true preserve_interword_spaces=1 textord_noise_rejwords=1 tessedit_adapt_to_char_fragments=0 user_defined_dpi=300 load_system_dawg=false	load_freq_dawg=false'
        # text = ftfy.fix_text(pytesseract.image_to_string(img, lang='eng', config=custom_config))
        # print(text,"ye text hai ",type(text))
        ocr = PaddleOCR(use_angle_cls=True, lang="en" ,use_mp=False,  cpu_threads=4,det_model_dir='models/ch_PP-OCRv3_det_infer', rec_model_dir='models/ch_PP-OCRv3_rec_infer')
        results = ocr.ocr(save_path, cls=True)

        # Convert OCR results into a single text block
        text = "\n".join([line[1][0] for result in results for line in result])
        
        # print(text,"yeocrtexthai=")
        extracted_data = extract_pan_data(text)
        
        if not extracted_data.get("ocr_name"):
            extracted_data=extract_old_pan_data(text)
            
        
       

        extracted_data['ocr_dob'] = normalize_dob_format(extracted_data['ocr_dob'])

        db_values = {
            'name': name,
            'pan': pan,
            'father_name': father_name,
            'dob': dob
        }

        matching_results = {
            'name_match_percentage': round(string_matching_percentage(extracted_data['ocr_name'].replace(" ", "").lower(), db_values['name'].replace(" ", "").lower()),2),
            'father_name_match_percentage':round(string_matching_percentage(extracted_data['ocr_father_name'].replace(" ", "").lower(), db_values['father_name'].replace(" ", "").lower()),2),
            'dob_match_percentage': round(string_matching_percentage(extracted_data['ocr_dob'].replace(" ", "").lower(), db_values['dob'].replace(" ", "").lower()),2),
            'pan_number_match_percentage': round(string_matching_percentage(extracted_data['ocr_pan_number'].replace(" ", "").lower(), db_values['pan'].replace(" ", "").lower()),2)
        }
        if extracted_data["ocr_father_name"]:
            total=int(matching_results["name_match_percentage"])+int(matching_results["father_name_match_percentage"])+int(matching_results["dob_match_percentage"])+int(matching_results["pan_number_match_percentage"])
            average_match_percentage=int(total/4)
            
        else:
            total=int(matching_results["name_match_percentage"])+int(matching_results["dob_match_percentage"])+int(matching_results["pan_number_match_percentage"])
           
            average_match_percentage=int(total/3)
            
            
        response = {
            'extracted_data': extracted_data,
            'matching_results': matching_results,
            'database_values': db_values,
            "average_match_percentage":average_match_percentage
        }

        
        
        
        response_data = {
        "status": "success",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "Processed successfully",
        "data":{"matching_results":matching_results} ,
        "req_id":req_id,
        "sources" : sources,
        "error" :""   
                
        }
        
        
        pandata={"name":name,"pan":pan,"father_name":father_name,"dob":dob} 
        pandata=json.dumps(pandata)
        conn, cursor = connectDB()
        
        
        # company=data["company"]
        resdata={"matching_results":matching_results,"extracted_data":extracted_data}
        # resdata=str(resdata)
        resdata = json.dumps(resdata)  
        cursor.execute("INSERT INTO ocrdetails(company_name,request_data, source,requested_for,response_data,status,request_id,company_id) VALUES (%s, %s, %s, %s,%s,%s,%s,%s)",(company, pandata,sources,"pan",resdata,"Completed",req_id,token))
        
        
        conn.commit()
        
        logger.info(f"response_data_extract_pan_details ",extra={"response_data":response_data})
        
        
        
        
        # log_entry = {
        # "request": request_data,
        # "response": response_data
        # }
        
        
        # savelogs(log_entry,log_file_path)
        cursor.execute(" select credits_left  from companydetails where  company_id=%s ",(token,))
        data1 =cursor.fetchall()
        credits_left=data1[0][0]
        credits_left=int(credits_left)-1
        cursor.execute(" update companydetails set  credits_left=%s where  company_id=%s ",(credits_left,token))
        conn.commit()
        
        
        
        
        return jsonify({"matching_results":matching_results,"extracted_data":extracted_data,"average_match_percentage":average_match_percentage,"success":True}), 200

    except Exception as e:
        # request_data = {
        # "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # "headers": dict(request.headers),
        # "form_data": request.form.to_dict(),
        # "json_data": request.get_json() if request.is_json else None,
        # "files": list(request.files.keys())  # Only logs file names
        #     }
        
        conn, cursor = connectDB()
        
        
        # company=data["company"]
        pandata={"name":name,"pan":pan,"father_name":father_name,"dob":dob}
        pandata=json.dumps(pandata)
        resdata={}
        resdata=json.dumps(resdata)
        cursor.execute("INSERT INTO ocrdetails(company_name,request_data, source,requested_for,response_data,status,request_id) VALUES (%s, %s, %s, %s,%s,%s,%s)",(company, pandata,sources,"adhaar",resdata,"Failed",req_id))
        
        
        conn.commit()
        
        response_data = {
        "status": "fail",
        "message": "Processed successfully",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data":"" ,
        "error" :str(e) ,
        "req_id":req_id,
        "sources" : sources  
                
        }
        logger.info(f"Error_response_data_extract_pan_details ",exc_info=True,extra={"response_data":response_data})
        # log_entry = {
        # "request": request_data,
        # "response": response_data
        # }
        
        # savelogs(log_entry,log_file_path)
        
        traceback.print_exc()
        return jsonify({'msg': str(e),"success":False}), 500
    
    
@app.route('/extract_pan_details_byurl', methods=['POST'])
# @jwt_required() 
def extract_pan_details_byurl():
    print("Files received:", list(request.files.keys()))
    data = request.get_json()
    
    file = data['panfile']
    name = data.get('name')
    pan = data.get('pan')
    sources = data.get('sources')
    father_name = data.get('fathername')
    dob = data.get('dob')
    req_id = data.get('req_id')
    company = data.get('company')
    dob=convert_to_ddmmyyyy(dob)
    # token=get_jwt_identity()
    token="19"

    sta,msg=check_auth(token,"pan_access")
    if not sta:
        return jsonify({'msg':msg ,"success":False})
    try:
        
    #     if 'clientcode' not in request.json:
    #         return jsonify({'msg': 'clientcode is required'}), 400

    #     clientcode = request.json['clientcode']
        
        
        
        
        
        request_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "headers": dict(request.headers),
        
        "files": list(request.files.keys()) ,
        "req_id":req_id,
        "sources" : sources
        
        
            }
        
        logger.info(f"request_data_extract_pan_details ",extra=request_data)
        
        
        
        if not sources:
            return {"msg":"Please provide sources","success":False}
        
        path=f"static/panfiles/{sources}"
        
        pathlog=f"static/logs/panfiles/{sources}"
        if not os.path.exists(pathlog):
            os.makedirs(pathlog)
            
        log_file_path = os.path.join(pathlog, f"{pan}.json")
            
        if not os.path.exists(path):
            os.makedirs(path)
            
        
        response = requests.get(file, stream=True)
        if response.status_code != 200:
            return {"success": False, "msg": "Failed to download file"}
        
        
        file_content = BytesIO(response.content)  # Read file into memory
        file_content.seek(0)  # Reset pointer
        new_filename=f"{pan}.jpg"
        original_extension = os.path.splitext(file.split("/")[-1].split("?")[0])[-1].lower()
        save_path1 = os.path.join(path, f"{pan}{original_extension}")
        save_path = os.path.join(path, f"{pan}.jpg")
        original_extension = os.path.splitext(file.split("/")[-1].split("?")[0])[-1].lower()
        content_type = response.headers.get("Content-Type", "").lower()
        
        
        if "pdf" in content_type or file.lower().endswith(".pdf"):
                file = FileStorage(stream=file_content, filename=new_filename + ".pdf", content_type="application/pdf")
                
                convert_pdf_to_image(file, save_path)
                
        elif "image"  in content_type :
            original_extension=".jpg"
            save_path1 = os.path.join(path, f"{pan}{original_extension}")
            
            with open(save_path1, "wb") as f:
                f.write(file_content.getbuffer())  
    
        
        
        
        
        
        # new_filename = f"{pan}.jpg"  # You can adjust the file extension as needed
        # extension = os.path.splitext(file.filename)[1]
        
        # new_filename = f"{pan}{extension}"
        # new_filename1=f"{pan}_processed{extension}"
        # # Full path for saving the file
        # save_path = os.path.join(path, new_filename)
        # save_path1 = os.path.join(path, new_filename1)
        # # save_path = os.path.join(path, new_filename)
        
        # file.save(save_path)
  


        
        
        # img = Image.open(save_path)
        # preprocessed_img = preprocess_image(img)
        # preprocessed_img = np.array(preprocessed_img)
        # print(preprocessed_img,type(preprocessed_img),"nknkj")
        
 
        # reader = easyocr.Reader(['en'], gpu=True) 
        # text= reader.readtext(img, detail=0)
        # text="\n".join(text) 
        # img = Image.open(save_path).convert("L")
        
        # custom_config = r'--oem 1 --psm 6 -c textord_heavy_nr=1 tessedit_write_images=true preserve_interword_spaces=1 textord_noise_rejwords=1 tessedit_adapt_to_char_fragments=0 user_defined_dpi=300 load_system_dawg=false	load_freq_dawg=false'
        # text = ftfy.fix_text(pytesseract.image_to_string(img, lang='eng', config=custom_config))
        # print(text,"ye text hai ",type(text))
        conn, cursor = connectDB()
        ocr = PaddleOCR(use_angle_cls=True, lang="en" ,use_mp=False,  cpu_threads=4,det_model_dir='models/ch_PP-OCRv3_det_infer', rec_model_dir='models/ch_PP-OCRv3_rec_infer')
        results = ocr.ocr(save_path, cls=True)

        # Convert OCR results into a single text block
        text = "\n".join([line[1][0] for result in results for line in result])
        cursor.execute("INSERT INTO extract_data(source,request_id,requested_for,details) VALUES (%s, %s, %s,%s)",(sources,req_id,"pan",text))
        
        
        conn.commit()
        # print(text,"yeocrtexthai=")
        extracted_data = extract_pan_data(text)
        
        if not extracted_data.get("ocr_name"):
            extracted_data=extract_old_pan_data(text)
            
        
       

        extracted_data['ocr_dob'] = normalize_dob_format(extracted_data['ocr_dob'])

        db_values = {
            'name': name,
            'pan': pan,
            'father_name': father_name,
            'dob': dob
        }

        matching_results = {
            'name_match_percentage':round( string_matching_percentage(extracted_data['ocr_name'].replace(" ", "").lower(), db_values['name'].replace(" ", "").lower()),2),
            'father_name_match_percentage': round(string_matching_percentage(extracted_data['ocr_father_name'].replace(" ", "").lower(), db_values['father_name'].replace(" ", "").lower()),2),
            'dob_match_percentage': round(string_matching_percentage(extracted_data['ocr_dob'].replace(" ", "").lower(), db_values['dob'].replace(" ", "").lower()),2),
            'pan_number_match_percentage': round(string_matching_percentage(extracted_data['ocr_pan_number'].replace(" ", "").lower(), db_values['pan'].replace(" ", "").lower()),2)
        }
        if extracted_data["ocr_father_name"]:
            total=int(matching_results["name_match_percentage"])+int(matching_results["father_name_match_percentage"])+int(matching_results["dob_match_percentage"])+int(matching_results["pan_number_match_percentage"])
            average_match_percentage=int(total/4)
            
        else:
            total=int(matching_results["name_match_percentage"])+int(matching_results["dob_match_percentage"])+int(matching_results["pan_number_match_percentage"])
           
            average_match_percentage=int(total/3)
        response = {
            'extracted_data': extracted_data,
            'matching_results': matching_results,
            'database_values': db_values,
            'average_match_percentage':average_match_percentage
        }

        
        
        
        response_data = {
        "status": "success",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "Processed successfully",
        "data":{"matching_results":matching_results} ,
        "req_id":req_id,
        "sources" : sources,
        "error" :""   
                
        }
        
        
        pandata={"name":name,"pan":pan,"father_name":father_name,"dob":dob} 
        pandata=json.dumps(pandata)
       
        
        
        # company=data["company"]
        resdata={"matching_results":matching_results,"extracted_data":extracted_data}
        # resdata=str(resdata)
        resdata = json.dumps(resdata)  
        cursor.execute("INSERT INTO ocrdetails(company_name,request_data, source,requested_for,response_data,status,request_id,company_id) VALUES (%s, %s, %s, %s,%s,%s,%s,%s)",(company, pandata,sources,"pan",resdata,"Completed",req_id,token))
        
        
        conn.commit()
        
        logger.info(f"response_data_extract_pan_details ",extra={"response_data":response_data})
        
        
        
        
        # log_entry = {
        # "request": request_data,
        # "response": response_data
        # }
        
        
        # savelogs(log_entry,log_file_path)
        cursor.execute(" select credits_left  from companydetails where  company_id=%s ",(token,))
        data1 =cursor.fetchall()
        credits_left=data1[0][0]
        credits_left=int(credits_left)-1
        cursor.execute(" update companydetails set  credits_left=%s where  company_id=%s ",(credits_left,token))
        conn.commit()
        
        
        
        
        return jsonify({"matching_results":matching_results,"extracted_data":extracted_data,"average_match_percentage":average_match_percentage,"success":True}), 200

    except Exception as e:
        # request_data = {
        # "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # "headers": dict(request.headers),
        # "form_data": request.form.to_dict(),
        # "json_data": request.get_json() if request.is_json else None,
        # "files": list(request.files.keys())  # Only logs file names
        #     }
        
        conn, cursor = connectDB()
        
        
        # company=data["company"]
        pandata={"name":name,"pan":pan,"father_name":father_name,"dob":dob}
        pandata=json.dumps(pandata)
        resdata={}
        resdata=json.dumps(resdata)
        cursor.execute("INSERT INTO ocrdetails(company_name,request_data, source,requested_for,response_data,status,request_id) VALUES (%s, %s, %s, %s,%s,%s,%s)",(company, pandata,sources,"adhaar",resdata,"Failed",req_id))
        
        
        conn.commit()
        
        response_data = {
        "status": "fail",
        "message": "Processed successfully",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data":"" ,
        "error" :str(e) ,
        "req_id":req_id,
        "sources" : sources  
                
        }
        logger.info(f"Error_response_data_extract_pan_details ",exc_info=True,extra={"response_data":response_data})
        # log_entry = {
        # "request": request_data,
        # "response": response_data
        # }
        
        # savelogs(log_entry,log_file_path)
        
        traceback.print_exc()
        return jsonify({'msg': str(e),"success":False}), 500
    
    

@app.route('/extract_adhar_details', methods=['POST'])
@jwt_required()  
def extract_adhar_details():
    
    
    print("Files received:", list(request.files.keys()))
    token=get_jwt_identity()
    # conn, cursor = connectDB()
    # cursor.execute(" select password  from companydetails where  company_id=%s ",(id,))
    # data1 =cursor.fetchall()
    sta,msg=check_auth(token,"aadhar_access")
    if not sta:
        return jsonify({'msg':msg ,"success":False})
    
    
    file = request.files['adharfile']
    name = request.form.get('name')
    adharno = request.form.get('adharno')
    sources = request.form.get('sources')
    address = request.form.get('address')
    dob = request.form.get('dob')
    req_id = request.form.get('req_id')
    company = request.form.get('company')
    dob=convert_to_ddmmyyyy(dob)
    try:
    #     if 'clientcode' not in request.json:
    #         return jsonify({'msg': 'clientcode is required'}), 400

    #     clientcode = request.json['clientcode']
        
        
        
        
        
        request_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "headers": dict(request.headers),
        
        "files": list(request.files.keys()) ,
        "req_id":req_id,
        "sources" : sources
        
        
            }
        
        logger.info(f"request_data_extract_adhar_details ",extra=request_data)
        
        
        
        if not sources:
            return {"msg":"Please provide sources","success":False}
        
        
        
        
        
        path=f"static/adharfiles/{sources}"
        
        pathlog=f"static/logs/adharfiles/{sources}"
        if not os.path.exists(pathlog):
            os.makedirs(pathlog)
            
        log_file_path = os.path.join(pathlog, f"{adharno}.json")
            
        if not os.path.exists(path):
            os.makedirs(path)
            
        
        # new_filename = f"{pan}.jpg"  # You can adjust the file extension as needed
        extension = os.path.splitext(file.filename)[1]
        new_filename = f"{adharno}{extension}"
        new_filename1=f"{adharno}.jpg"
        # Full path for saving the file
        save_path = os.path.join(path, new_filename)
        save_path1 = os.path.join(path, new_filename1)
        # save_path = os.path.join(path, new_filename)
        if extension.lower()==".pdf":
            convert_pdf_to_image(file,save_path1)
            save_path=save_path1
            
        else:   
            file.save(save_path)
  


        
        
        # img = Image.open(save_path)
        # preprocessed_img = preprocess_image(img)
        # preprocessed_img = np.array(preprocessed_img)
        # print(preprocessed_img,type(preprocessed_img),"nknkj")
        
 
        # reader = easyocr.Reader(['en'], gpu=True) 
        # text= reader.readtext(img, detail=0)
        # text="\n".join(text) 
        # img = Image.open(save_path).convert("L")
        
        # custom_config = r'--oem 1 --psm 6 -c textord_heavy_nr=1 tessedit_write_images=true preserve_interword_spaces=1 textord_noise_rejwords=1 tessedit_adapt_to_char_fragments=0 user_defined_dpi=300 load_system_dawg=false	load_freq_dawg=false'
        # text = ftfy.fix_text(pytesseract.image_to_string(img, lang='eng', config=custom_config))
        # print(text,"ye text hai ",type(text))
        ocr = PaddleOCR(use_angle_cls=True, lang="en" ,use_mp=False,  cpu_threads=4,det_model_dir='models/ch_PP-OCRv3_det_infer', rec_model_dir='models/ch_PP-OCRv3_rec_infer')
        results = ocr.ocr(save_path, cls=True)

        # Convert OCR results into a single text block
        text = "\n".join([line[1][0] for result in results for line in result])
        print(text)
        
        conn, cursor = connectDB()
        
        
        # company=data["company"]
        resdata={}
        cursor.execute("INSERT INTO extract_data(source,request_id,requested_for,details) VALUES (%s, %s, %s,%s)",(sources,req_id,"adhaar",text))
        
        
        conn.commit()
        # print(text,"yeocrtexthai=")
        
        extracted_data=DigiLockeraadhar(text)
        print(extracted_data,"DigiLockeraadhar")
        if not extracted_data.get("ocr_name") or  not extracted_data.get("ocr_address")  or  not  extracted_data['ocr_dob'] :
         extracted_data =  extract_full_page_adahar(text)
        
        if not extracted_data.get("ocr_name") or not extracted_data.get("ocr_address")  :
            
            extracted_data=extract_both_side_adahar(text)
            
        # if not extracted_data.get("ocr_name") and  not extracted_data.get("ocr_address")  :
        #       extracted_data=extract_both_side_reissue_adahar(text)
              
                 
            
        
        if not extracted_data.get("ocr_name") :
            
            extracted_data=extract_front_page_adahar(text)
        
        # if not extracted_data.get("ocr_name"):
        #     extracted_data=extract_front_page_reissue_adahar(text)
                
        
       

        extracted_data['ocr_dob'] = normalize_dob_format(extracted_data['ocr_dob'])

        db_values = {
            'name': name,
            'adharno': adharno,
            'address': address,
            'dob': dob
        }
       
        matching_results = {
            'name_match_percentage': round(string_matching_percentage(extracted_data['ocr_name'].replace(" ", "").lower(), db_values['name'].replace(" ", "").lower()),2),
            'ocr_address_match_percentage': round(string_matching_percentage(extracted_data['ocr_address_for_match'].replace(" ", "").lower(), db_values['address'].replace(" ", "").lower()),2),
            'dob_match_percentage':round( string_matching_percentage(extracted_data['ocr_dob'].replace(" ", "").lower(), db_values['dob'].replace(" ", "").lower()),2),
            'adharno_number_match_percentage': round(string_matching_percentage(extracted_data['ocr_adhar_number'].replace(" ", "").lower(), db_values['adharno'].replace(" ", "").lower()),)
        }
        if extracted_data["ocr_address_for_match"]:
            total=int(matching_results["name_match_percentage"])+int(matching_results["ocr_address_match_percentage"])+int(matching_results["dob_match_percentage"])+int(matching_results["adharno_number_match_percentage"])
            average_match_percentage=int(total/4)
            
        else:
            total=int(matching_results["name_match_percentage"])+int(matching_results["dob_match_percentage"])+int(matching_results["adharno_number_match_percentage"])
           
            average_match_percentage=int(total/3)

        response = {
            'extracted_data': extracted_data,
            'matching_results': matching_results,
            'database_values': db_values,
            "average_match_percentage":average_match_percentage
        }

        
        
        
        response_data = {
        "status": "success",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "Processed successfully",
        "data":{"matching_results":matching_results} ,
        "req_id":req_id,
        "sources" : sources,
        "error" :""   
                
        }
        
        logger.info(f"response_data_extract_adhar_details ",extra={"response_data":response_data})
        
        
        
        
        # log_entry = {
        # "request": request_data,
        # "response": response_data
        # }
        
        
        # savelogs(log_entry,log_file_path)
        adahardata={"name":name,"adharno":adharno,"address":address,"dob":dob} 
        adahardata=json.dumps(adahardata)
        conn, cursor = connectDB()
        
        
        # company=data["company"]
        resdata={"matching_results":matching_results,"extracted_data":extracted_data}
        # resdata=str(resdata)
        resdata = json.dumps(resdata)  
        cursor.execute("INSERT INTO ocrdetails(company_name,request_data, source,requested_for,response_data,status,request_id,company_id) VALUES (%s, %s, %s, %s,%s,%s,%s,%s)",(company, adahardata,sources,"adhaar",resdata,"Completed",req_id,token))
        
        
        conn.commit()
        
        
        cursor.execute(" select credits_left  from companydetails where  company_id=%s ",(token,))
        data1 =cursor.fetchall()
        credits_left=data1[0][0]
        credits_left=int(credits_left)-1
        cursor.execute(" update companydetails set  credits_left=%s where  company_id=%s ",(credits_left,token))
        conn.commit()
        
        
        
        return jsonify({"matching_results":matching_results,"extracted_data":extracted_data,"average_match_percentage":average_match_percentage,"success":True}), 200

    except Exception as e:
        # request_data = {
        # "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # "headers": dict(request.headers),
        # "form_data": request.form.to_dict(),
        # "json_data": request.get_json() if request.is_json else None,
        # "files": list(request.files.keys())  # Only logs file names
        #     }
        
        conn, cursor = connectDB()
        
        
        # company=data["company"]
        adahardata={"name":name,"adharno":adharno,"address":address,"dob":dob}
        adahardata=json.dumps(adahardata)
        resdata={}
        resdata=json.dumps(resdata)
        cursor.execute("INSERT INTO ocrdetails(company_name,request_data, source,requested_for,response_data,status,request_id) VALUES (%s, %s, %s, %s,%s,%s,%s)",(company, adahardata,sources,"adhaar",resdata,"Failed",req_id))
        
        
        conn.commit()
        
        response_data = {
        "status": "fail",
        "message": "Processed successfully",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data":"" ,
        "error" :str(e) ,
        "req_id":req_id,
        "sources" : sources  
                
        }
        logger.info(f"Error_response_data_extract_adhar_details ",exc_info=True,extra={"response_data":response_data})
        # log_entry = {
        # "request": request_data,
        # "response": response_data
        # }
        
        # savelogs(log_entry,log_file_path)
        
        traceback.print_exc()
        return jsonify({'msg': str(e),"success":False}), 500
   
   
@app.route('/extract_adhar_details_byurl', methods=['POST'])
# @jwt_required()  
def extract_adhar_details_byurl():
    
    
    # print("Files received:", list(request.files.keys()))
    # token=get_jwt_identity()
    token="19"
    # conn, cursor = connectDB()
    # cursor.execute(" select password  from companydetails where  company_id=%s ",(id,))
    # data1 =cursor.fetchall()
    data = request.get_json()
    sta,msg=check_auth(token,"aadhar_access")
    if not sta:
        return jsonify({'msg':msg ,"success":False})
    
    
    file =data['adharfile']
    name = data.get('name')
    adharno = data.get('adharno')
    sources = data.get('sources')
    address =data.get('address')
    dob = data.get('dob')
    req_id = data.get('req_id')
    company = data.get('company')
    dob=convert_to_ddmmyyyy(dob)
    try:
    #     if 'clientcode' not in request.json:
    #         return jsonify({'msg': 'clientcode is required'}), 400

    #     clientcode = request.json['clientcode']
        
        
        
        
        
        request_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "headers": dict(request.headers),
        
        "files": list(request.files.keys()) ,
        "req_id":req_id,
        "sources" : sources
        
        
            }
        
        logger.info(f"request_data_extract_adhar_details ",extra=request_data)
        
        
        
        if not sources:
            return {"msg":"Please provide sources","success":False}
        
        
        
        
        
        path=f"static/adharfiles/{sources}"
        
        pathlog=f"static/logs/adharfiles/{sources}"
        if not os.path.exists(pathlog):
            os.makedirs(pathlog)
            
        log_file_path = os.path.join(pathlog, f"{adharno}.json")
            
        if not os.path.exists(path):
            os.makedirs(path)
            
        
        response = requests.get(file, stream=True)
        if response.status_code != 200:
            return {"success": False, "msg": "Failed to download file"}
        
        
        file_content = BytesIO(response.content)  # Read file into memory
        file_content.seek(0)  # Reset pointer
        new_filename=f"{adharno}.jpg"
        original_extension = os.path.splitext(file.split("/")[-1].split("?")[0])[-1].lower()
        save_path1 = os.path.join(path, f"{adharno}{original_extension}")
        save_path = os.path.join(path, f"{adharno}.jpg")
        original_extension = os.path.splitext(file.split("/")[-1].split("?")[0])[-1].lower()
        content_type = response.headers.get("Content-Type", "").lower()
        
        
        if "pdf" in content_type or file.lower().endswith(".pdf"):
                file = FileStorage(stream=file_content, filename=new_filename + ".pdf", content_type="application/pdf")
                
                convert_pdf_to_image(file, save_path)
                
        elif "image"  in content_type :
            original_extension=".jpg"
            save_path1 = os.path.join(path, f"{adharno}{original_extension}")
            
            with open(save_path1, "wb") as f:
                f.write(file_content.getbuffer())        
        
        # extension = os.path.splitext(file.filename)[1]
        # new_filename = f"{adharno}{extension}"
        
  


        
        
        # img = Image.open(save_path)
        # preprocessed_img = preprocess_image(img)
        # preprocessed_img = np.array(preprocessed_img)
        # print(preprocessed_img,type(preprocessed_img),"nknkj")
        
 
        # reader = easyocr.Reader(['en'], gpu=True) 
        # text= reader.readtext(img, detail=0)
        # text="\n".join(text) 
        # img = Image.open(save_path).convert("L")
        
        # custom_config = r'--oem 1 --psm 6 -c textord_heavy_nr=1 tessedit_write_images=true preserve_interword_spaces=1 textord_noise_rejwords=1 tessedit_adapt_to_char_fragments=0 user_defined_dpi=300 load_system_dawg=false	load_freq_dawg=false'
        # text = ftfy.fix_text(pytesseract.image_to_string(img, lang='eng', config=custom_config))
        # print(text,"ye text hai ",type(text))
        ocr = PaddleOCR(use_angle_cls=True, lang="en" ,use_mp=False,  cpu_threads=4,det_model_dir='models/ch_PP-OCRv3_det_infer', rec_model_dir='models/ch_PP-OCRv3_rec_infer')
        results = ocr.ocr(save_path, cls=True)

        # Convert OCR results into a single text block
        text = "\n".join([line[1][0] for result in results for line in result])
        print(text)
        
        conn, cursor = connectDB()
        
        
        # company=data["company"]
        resdata={}
        cursor.execute("INSERT INTO extract_data(source,request_id,requested_for,details) VALUES (%s, %s, %s,%s)",(sources,req_id,"adhaar",text))
        
        
        conn.commit()
        # print(text,"yeocrtexthai=")
        
        extracted_data=DigiLockeraadhar(text)
        print(extracted_data,"DigiLockeraadhar")
        if not extracted_data.get("ocr_name") or  not extracted_data.get("ocr_address")  or  not  extracted_data['ocr_dob'] :
         extracted_data =  extract_full_page_adahar(text)
        
        if not extracted_data.get("ocr_name") or not extracted_data.get("ocr_address")  :
            
            extracted_data=extract_both_side_adahar(text)
            
        # if not extracted_data.get("ocr_name") and  not extracted_data.get("ocr_address")  :
        #       extracted_data=extract_both_side_reissue_adahar(text)
              
                 
            
        
        if not extracted_data.get("ocr_name") :
            
            extracted_data=extract_front_page_adahar(text)
        
        # if not extracted_data.get("ocr_name"):
        #     extracted_data=extract_front_page_reissue_adahar(text)
                
        
       

        extracted_data['ocr_dob'] = normalize_dob_format(extracted_data['ocr_dob'])

        db_values = {
            'name': name,
            'adharno': adharno,
            'address': address,
            'dob': dob
        }
        
        matching_results = {
            'name_match_percentage': round(string_matching_percentage(extracted_data['ocr_name'].replace(" ", "").lower(), db_values['name'].replace(" ", "").lower()),2),
            'ocr_address_match_percentage': round(string_matching_percentage(extracted_data['ocr_address_for_match'].replace(" ", "").lower(), db_values['address'].replace(" ", "").lower()),2),
            'dob_match_percentage': round(string_matching_percentage(extracted_data['ocr_dob'].replace(" ", "").lower(), db_values['dob'].replace(" ", "").lower()),2),
            'adharno_number_match_percentage': round(string_matching_percentage(extracted_data['ocr_adhar_number'].replace(" ", "").lower(), db_values['adharno'].replace(" ", "").lower()),2)
        }
        if extracted_data["ocr_address_for_match"]:
            total=int(matching_results["name_match_percentage"])+int(matching_results["ocr_address_match_percentage"])+int(matching_results["dob_match_percentage"])+int(matching_results["adharno_number_match_percentage"])
            average_match_percentage=int(total/4)
            
        else:
            total=int(matching_results["name_match_percentage"])+int(matching_results["dob_match_percentage"])+int(matching_results["adharno_number_match_percentage"])
           
            average_match_percentage=int(total/3)
        response = {
            'extracted_data': extracted_data,
            'matching_results': matching_results,
            'database_values': db_values,
            'average_match_percentage':average_match_percentage
        }

        
        
        
        response_data = {
        "status": "success",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "Processed successfully",
        "data":{"matching_results":matching_results} ,
        "req_id":req_id,
        "sources" : sources,
        "error" :""   
                
        }
        
        logger.info(f"response_data_extract_adhar_details ",extra={"response_data":response_data})
        
        
        
        
        # log_entry = {
        # "request": request_data,
        # "response": response_data
        # }
        
        
        # savelogs(log_entry,log_file_path)
        adahardata={"name":name,"adharno":adharno,"address":address,"dob":dob} 
        adahardata=json.dumps(adahardata)
        conn, cursor = connectDB()
        
        
        # company=data["company"]
        resdata={"matching_results":matching_results,"extracted_data":extracted_data}
        # resdata=str(resdata)
        resdata = json.dumps(resdata)  
        cursor.execute("INSERT INTO ocrdetails(company_name,request_data, source,requested_for,response_data,status,request_id,company_id) VALUES (%s, %s, %s, %s,%s,%s,%s,%s)",(company, adahardata,sources,"adhaar",resdata,"Completed",req_id,token))
        
        
        conn.commit()
        
        
        cursor.execute(" select credits_left  from companydetails where  company_id=%s ",(token,))
        data1 =cursor.fetchall()
        credits_left=data1[0][0]
        credits_left=int(credits_left)-1
        cursor.execute(" update companydetails set  credits_left=%s where  company_id=%s ",(credits_left,token))
        conn.commit()
        
        
        
        return jsonify({"matching_results":matching_results,"extracted_data":extracted_data,"average_match_percentage":average_match_percentage,"success":True}), 200

    except Exception as e:
        # request_data = {
        # "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # "headers": dict(request.headers),
        # "form_data": request.form.to_dict(),
        # "json_data": request.get_json() if request.is_json else None,
        # "files": list(request.files.keys())  # Only logs file names
        #     }
        
        conn, cursor = connectDB()
        
        
        # company=data["company"]
        adahardata={"name":name,"adharno":adharno,"address":address,"dob":dob}
        adahardata=json.dumps(adahardata)
        resdata={}
        resdata=json.dumps(resdata)
        cursor.execute("INSERT INTO ocrdetails(company_name,request_data, source,requested_for,response_data,status,request_id) VALUES (%s, %s, %s, %s,%s,%s,%s)",(company, adahardata,sources,"adhaar",resdata,"Failed",req_id))
        
        
        conn.commit()
        
        response_data = {
        "status": "fail",
        "message": "Processed successfully",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data":"" ,
        "error" :str(e) ,
        "req_id":req_id,
        "sources" : sources  
                
        }
        logger.info(f"Error_response_data_extract_adhar_details ",exc_info=True,extra={"response_data":response_data})
        # log_entry = {
        # "request": request_data,
        # "response": response_data
        # }
        
        # savelogs(log_entry,log_file_path)
        
        traceback.print_exc()
        return jsonify({'msg': str(e),"success":False}), 500
       
   





 
@app.route('/extract_financial_details', methods=['POST'])
def extract_financial_details():
    try:
        # Extract data from the query result
        file = request.files['financialprooffile']
        name = request.form.get('name')
        sources = request.form.get('sources')
        micr = request.form.get('micr')
        ifsc = request.form.get('ifsc')
        account_number = request.form.get('account_number')
        req_id = request.form.get('req_id')
        
        
        request_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "headers": dict(request.headers),
        
        "files": list(request.files.keys()) ,
        "req_id":req_id,
        "sources" : sources
        
        
            }
        
        logger.info(f"request_data_extract_financial_details ",extra=request_data)
        
        if not sources:
            return {"msg":"Please provide sources","success":False}
        
        path=f"static/financialprooffile/{sources}"
        
        pathlog=f"static/logs/financialprooffile/{sources}"
        if not os.path.exists(pathlog):
            os.makedirs(pathlog)
        log_file_path = os.path.join(pathlog, f"{account_number}.json")   
        if not os.path.exists(path):
            os.makedirs(path)
        
        extension = os.path.splitext(file.filename)[1]

        # Create a new filename using the account number and the original extension
        new_filename = f"{account_number}{extension}"
        save_path = os.path.join(path, new_filename)
        # save_path = os.path.join(path, new_filename)
        file.save(save_path)
        
        
        # upload_financialproof_path, name, micr, ifsc, account_number 

        if not save_path or not os.path.exists(save_path):
            return jsonify({'msg': 'No valid financial proof path (uploadFinancialproof) found for the provided clientcode or file does not exist'}), 404

        try:
            content_type = ""
            text = ""

            if save_path.lower().endswith('.pdf'):
                try:
                    with pdfplumber.open(save_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                            
                    if not text.strip():
                        pdf_images = convert_from_bytes(open(save_path, 'rb').read())
                        for image in pdf_images:
                            text += pytesseract.image_to_string(image, lang='eng') + "\n"

                    text = clean_text(text)
                except Exception as e:
                    return jsonify({"msg": f"Failed to process the PDF file: {str(e)}"}), 400

            elif save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = Image.open(save_path)
                    text = pytesseract.image_to_string(img, lang='eng')
                    text = clean_text(text)
                except Exception as e:
                    return jsonify({"msg": f"Failed to process the image file: {str(e)}"}), 400

            else:
                return jsonify({"msg": "Unsupported file type. Only PDF and image files are supported."}), 400

        except Exception as e:
            return jsonify({"msg": f"Error processing the file: {str(e)}"}), 400

        extracted_data = extract_financial_data(text)

        db_values = {
            'Client Name': name,
            'MICR Code': micr,
            'IFSC Code': ifsc,
            'Account Number': account_number
        }

        matching_results = {
            'client_name_match_percentage': string_matching_percentage(extracted_data.get('Client Name'), db_values.get('Client Name')),
            'micr_match_percentage': string_matching_percentage(extracted_data.get('MICR Code'), db_values.get('MICR Code')),
            'ifsc_match_percentage': string_matching_percentage(extracted_data.get('IFSC Code'), db_values.get('IFSC Code')),
            'account_number_match_percentage': string_matching_percentage(extracted_data.get('Account Number'), db_values.get('Account Number'))
        }

        response = {
            'extracted_data': extracted_data,
            'database_values': db_values,
            'matching_results': matching_results
        }
        
        # request_data = {
        # "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # "headers": dict(request.headers),
        # "form_data": request.form.to_dict(),
        # "json_data": request.get_json() if request.is_json else None,
        # "files": list(request.files.keys())  # Only logs file names
        #     }
        
        
        response_data = {
        "status": "success",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "Processed successfully",
        "data":{"matching_results":matching_results} ,
        "req_id":req_id,
        "sources" : sources,
        "error" :""   
                
        }
        
        logger.info(f"response_data_extract_financial_details",extra={"response_data":response_data})
        
        
        # log_entry = {
        # "request": request_data,
        # "response": response_data
        # }
        
        
        # savelogs(log_entry,log_file_path)
        
        
        return jsonify({"matching_results":matching_results,"extracted_data":extracted_data,"success":False}), 200

    except Exception as e:
        # request_data = {
        # "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # "headers": dict(request.headers),
        # "form_data": request.form.to_dict(),
        # "json_data": request.get_json() if request.is_json else None,
        # "files": list(request.files.keys())  # Only logs file names
        #     }
        
        
        response_data = {
        "status": "fail",
        "message": "Processed successfully",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data":"" ,
        "error" :str(e) ,
        "req_id":req_id,
        "sources" : sources  
                
        }
        logger.info(f"Error_response_data_extract_financial_details", exc_info=True,extra={"response_data":response_data})
        
        # log_entry = {
        # "request": request_data,
        # "response": response_data
        # }
        
        # savelogs(log_entry,log_file_path)
        traceback.print_exc()
        
        print(f"Error: {e}")
        return jsonify({"msg": f"An error occurred while processing the request: {str(e)}","success":False}), 500

    

@app.route('/extract_bank_details', methods=['POST'])
def extract_bank_details():
    
    try:
   
    # try:
    #     if 'clientcode' not in request.json:
    #         return jsonify({"msg": "clientcode is required."}), 400

    #     clientcode = request.json['clientcode']

    #     try:
    #         conn = get_db_connection()
    #         cursor = conn.cursor()
    #         cursor.execute("""
    #             SELECT uploadBankproof, IFSC, MICR, name 
    #             FROM newuserdetails 
    #             WHERE clientcode = ?
    #         """, (clientcode,))
    #         db_data = cursor.fetchone()
    #         conn.close()
    #     except Exception as e:
    #         return jsonify({'msg': f'Database error: {e}'}), 500

    #     if not db_data:
    #         return jsonify({'msg': 'No matching records found for the provided clientcode'}), 404

    #     upload_bankproof_path, ifsc, micr, account_holder_name = db_data
    
        file = request.files['bankprooffile']
        name = request.form.get('name')
        sources = request.form.get('sources')
        micr = request.form.get('micr')
        ifsc = request.form.get('ifsc')
        account_number = request.form.get('account_number')
        req_id = request.form.get('req_id')
        request_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "headers": dict(request.headers),
        
        "files": list(request.files.keys()) ,
        "req_id":req_id,
        "sources" : sources
        
        
            }
        
        logger.info(f"request_data_extract_bank_details ",extra=request_data)
        
        
        if not sources:
            return {"msg":"Please provide sources","success":False}
        path=f"static/bankprooffile/{sources}"
        
        pathlog=f"static/logs/bankprooffile/{sources}"
        if not os.path.exists(pathlog):
            os.makedirs(pathlog)
        log_file_path = os.path.join(pathlog, f"{account_number}.json")
        
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        extension = os.path.splitext(file.filename)[1]

        # Create a new filename using the account number and the original extension
        new_filename = f"{account_number}{extension}"
        save_path = os.path.join(path, new_filename)
        # save_path = os.path.join(path, new_filename)
        file.save(save_path)
    
    


        if not save_path or not os.path.exists(save_path):
            return jsonify({'msg': 'No valid bank proof image path (uploadBankproof) found for the provided clientcode or file does not exist'}), 404

        try:
            
            text="" 
            if save_path.lower().endswith('.pdf'):
                try:
                    with pdfplumber.open(save_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    
                    if not text.strip():
                        pdf_images = convert_from_bytes(open(save_path, 'rb').read())
                        for image in pdf_images:
                            text += pytesseract.image_to_string(image, lang='eng') + "\n"

                    text = clean_text(text)
                except Exception as e:
                    return jsonify({"msg": f"Failed to process the PDF file: {str(e)}"}), 400
            
            
            elif save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = Image.open(save_path)
                    text = pytesseract.image_to_string(img, lang='eng')
                    text = clean_text(text)
                except Exception as e:
                    return jsonify({"msg": f"Failed to process the image file: {str(e)}"}), 400

            else:
                return jsonify({"msg": "Unsupported file type. Only PDF and image files are supported."}), 400
                
                
                
                    
                    
                
            # img = Image.open(save_path)
            # preprocessed_img = preprocess_image(img)
            # text = pytesseract.image_to_string(preprocessed_img, lang='eng')
            text = ftfy.fix_text(text)
        except Exception as e:
            return jsonify({"msg": f"Error processing the image: {str(e)}"}), 400

        print("OCR Text:", text)

        extracted_data = extract_bank_data(text)
        print("Extracted Data:", extracted_data)

        db_values = {
            'ifsc_code': ifsc,
            'micr_code': micr,
            'account_holder_name': name,
            'Account Number': account_number
            
        }

        matching_results = {
            'ifsc_match_percentage': string_matching_percentage(extracted_data.get('ocr_IFSC'), db_values.get('ifsc_code')),
            'micr_match_percentage': string_matching_percentage(extracted_data.get('ocr_MICR'), db_values.get('micr_code')),
            'account_holder_name_match_percentage': string_matching_percentage(extracted_data.get('ocr_Account_Holder_Name'), db_values.get('account_holder_name')),
            'account_number_match_percentage': string_matching_percentage(extracted_data.get('ocr_account_number'), db_values.get('Account Number'))
            
        }

        response = {
            'extracted_data': extracted_data,
            'database_values': db_values,
            'matching_results': matching_results
        }
        
        
        # request_data = {
        # "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # "headers": dict(request.headers),
        # "form_data": request.form.to_dict(),
        # "json_data": request.get_json() if request.is_json else None,
        # "files": list(request.files.keys())  # Only logs file names
        #     }
        
        
        response_data = {
        "status": "success",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "Processed successfully",
        "data":{"matching_results":matching_results} ,
        "req_id":req_id,
        "sources" : sources,
        "error" :""   
                
        }
        
        logger.info(f"response_data_extract_bank_details",extra={"response_data":response_data})
        
        
        # log_entry = {
        # "request": request_data,
        # "response": response_data
        # }
        
        
        # savelogs(log_entry,log_file_path)

        return jsonify({"matching_results":matching_results,"extracted_data":extracted_data}), 200

    except Exception as e:
        
        # request_data = {
        # "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # "headers": dict(request.headers),
        # "form_data": request.form.to_dict(),
        # "json_data": request.get_json() if request.is_json else None,
        # "files": list(request.files.keys())  # Only logs file names
        #     }
        
        
        response_data = {
        "status": "fail",
        "message": "Processed successfully",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data":"" ,
        "error" :str(e) ,
        "req_id":req_id,
        "sources" : sources  
                
        }
        logger.info(f"Error_response_data_extract_bank_details",exc_info=True,extra={"response_data":response_data})
        
        # log_entry = {
        # "request": request_data,
        # "response": response_data
        # }
        traceback.print_exc()
        # savelogs(log_entry,log_file_path)
        print(f"Error: {e}")
        return jsonify({"msg": f"An error occurred while processing the request: {str(e)}"}), 500





@app.route('/extract_bank_details_byurl', methods=['POST'])
def extract_bank_details_byurl():
    
    try:
   
    # try:
    #     if 'clientcode' not in request.json:
    #         return jsonify({"msg": "clientcode is required."}), 400

    #     clientcode = request.json['clientcode']

    #     try:
    #         conn = get_db_connection()
    #         cursor = conn.cursor()
    #         cursor.execute("""
    #             SELECT uploadBankproof, IFSC, MICR, name 
    #             FROM newuserdetails 
    #             WHERE clientcode = ?
    #         """, (clientcode,))
    #         db_data = cursor.fetchone()
    #         conn.close()
    #     except Exception as e:
    #         return jsonify({'msg': f'Database error: {e}'}), 500

    #     if not db_data:
    #         return jsonify({'msg': 'No matching records found for the provided clientcode'}), 404

    #     upload_bankproof_path, ifsc, micr, account_holder_name = db_data
    
        # file = request.files['bankprooffile']
        # name = request.form.get('name')
        # sources = request.form.get('sources')
        # micr = request.form.get('micr')
        # ifsc = request.form.get('ifsc')
        # account_number = request.form.get('account_number')
        # req_id = request.form.get('req_id')
        token="19"
        conn, cursor = connectDB()
        data = request.get_json()
        file =data['bankprooffile']
        name = data.get('name')
        account_number = data.get('account_number')
        sources = data.get('sources')
        ifsc =data.get('ifsc')
        micr = data.get('micr')
        req_id = data.get('req_id')
        company = data.get('company')
        request_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "headers": dict(request.headers),
        
        "files": list(request.files.keys()) ,
        "req_id":req_id,
        "sources" : sources
        
        
            }
        
        logger.info(f"request_data_extract_bank_details ",extra=request_data)
        
        # response = requests.get(file, stream=True)
        # if response.status_code != 200:
        #     return {"success": False, "msg": "Failed to download file"}
        
        
        
        bankdata={"name":name,"account_number":account_number,"ifsc":ifsc,"micr":micr} 
        bankdata=json.dumps(bankdata)
        if not sources:
            return {"msg":"Please provide sources","success":False}
        path=f"static/bankprooffile/{sources}"
        
        pathlog=f"static/logs/bankprooffile/{sources}"
        if not os.path.exists(pathlog):
            os.makedirs(pathlog)
        log_file_path = os.path.join(pathlog, f"{account_number}.json")
        
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        # extension = os.path.splitext(file.filename)[1]

        # Create a new filename using the account number and the original extension
        # new_filename = f"{account_number}{extension}"
        # save_path = os.path.join(path, new_filename)
        # save_path = os.path.join(path, new_filename)
        # file.save(save_path)
        response = requests.get(file, stream=True)
        if response.status_code != 200:
            return {"success": False, "msg": "Failed to download file"}
        
        file_content = BytesIO(response.content)  # Read file into memory
        file_content.seek(0)  # Reset pointer
        original_extension = os.path.splitext(file.split("/")[-1].split("?")[0])[-1].lower()
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" in content_type or file.lower().endswith(".pdf"):
            
            original_extension=".pdf"
        elif "image"  in content_type :
            original_extension=".jpg"
        else:
            original_extension=".jpg"
            
            
            
        
        save_path = os.path.join(path, f"{account_number}{original_extension}")
        
        with open(save_path, "wb") as f:
                f.write(file_content.getbuffer())
    
    


        if not save_path or not os.path.exists(save_path):
            return jsonify({'msg': 'No valid bank proof image path (uploadBankproof) found for the provided clientcode or file does not exist'}), 404

        try:
            
            text="" 
            if save_path.lower().endswith('.pdf'):
                try:
                    with pdfplumber.open(save_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    
                    if not text.strip():
                        pdf_images = convert_from_bytes(open(save_path, 'rb').read())
                        for image in pdf_images:
                            text += pytesseract.image_to_string(image, lang='eng') + "\n"

                    text = clean_text(text)
                except Exception as e:
                    return jsonify({"msg": f"Failed to process the PDF file: {str(e)}"}), 400
            
            
            elif save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = Image.open(save_path)
                    text = pytesseract.image_to_string(img, lang='eng')
                    text = clean_text(text)
                except Exception as e:
                    return jsonify({"msg": f"Failed to process the image file: {str(e)}"}), 400

            else:
                return jsonify({"msg": "Unsupported file type. Only PDF and image files are supported."}), 400
                
                
                
                    
                    
                
            # img = Image.open(save_path)
            # preprocessed_img = preprocess_image(img)
            # text = pytesseract.image_to_string(preprocessed_img, lang='eng')
            text = ftfy.fix_text(text)
        except Exception as e:
            return jsonify({"msg": f"Error processing the image: {str(e)}"}), 400

        print("OCR Text:", text)
        cursor.execute("INSERT INTO extract_data(source,request_id,requested_for,details) VALUES (%s, %s, %s,%s)",(sources,req_id,"adhaar",text))
        conn.commit()
        extracted_data = extract_bank_data(text)
        print("Extracted Data:", extracted_data)
        
        

        db_values = {
            'ifsc_code': ifsc,
            'micr_code': micr,
            'account_holder_name': name,
            'Account Number': account_number
            
        }

        matching_results = {
            'ifsc_match_percentage': string_matching_percentage(extracted_data.get('ocr_IFSC'), db_values.get('ifsc_code')),
            'micr_match_percentage': string_matching_percentage(extracted_data.get('ocr_MICR'), db_values.get('micr_code')),
            'account_holder_name_match_percentage': string_matching_percentage(extracted_data.get('ocr_Account_Holder_Name'), db_values.get('account_holder_name')),
            'account_number_match_percentage': string_matching_percentage(extracted_data.get('ocr_account_number'), db_values.get('Account Number'))
            
        }
        
        total=int(matching_results["ifsc_match_percentage"])+int(matching_results["micr_match_percentage"])+int(matching_results["account_holder_name_match_percentage"])+int(matching_results["account_number_match_percentage"])
           
        average_match_percentage=int(total/4)

        response = {
            'extracted_data': extracted_data,
            'database_values': db_values,
            'matching_results': matching_results,
            'average_match_percentage':average_match_percentage
        }
        resdata={"matching_results":matching_results,"extracted_data":extracted_data}
        resdata = json.dumps(resdata) 
        cursor.execute("INSERT INTO ocrdetails(company_name,request_data, source,requested_for,response_data,status,request_id,company_id) VALUES (%s, %s, %s, %s,%s,%s,%s,%s)",(company, bankdata,sources,"adhaar",resdata,"Completed",req_id,token))
        
        
        conn.commit()
        
        
        
        
        
        cursor.execute(" select credits_left  from companydetails where  company_id=%s ",(token,))
        data1 =cursor.fetchall()
        credits_left=data1[0][0]
        credits_left=int(credits_left)-1
        cursor.execute(" update companydetails set  credits_left=%s where  company_id=%s ",(credits_left,token))
        conn.commit()
        
        # request_data = {
        # "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # "headers": dict(request.headers),
        # "form_data": request.form.to_dict(),
        # "json_data": request.get_json() if request.is_json else None,
        # "files": list(request.files.keys())  # Only logs file names
        #     }
        
        
        response_data = {
        "status": "success",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "Processed successfully",
        "data":{"matching_results":matching_results} ,
        "req_id":req_id,
        "sources" : sources,
        "error" :""   
                
        }
        
        logger.info(f"response_data_extract_bank_details",extra={"response_data":response_data})
        
        
        # log_entry = {
        # "request": request_data,
        # "response": response_data
        # }
        
        
        # savelogs(log_entry,log_file_path)

        return jsonify({"matching_results":matching_results,"extracted_data":extracted_data,"average_match_percentage":average_match_percentage,"success":True}), 200

    except Exception as e:
        
        # request_data = {
        # "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # "headers": dict(request.headers),
        # "form_data": request.form.to_dict(),
        # "json_data": request.get_json() if request.is_json else None,
        # "files": list(request.files.keys())  # Only logs file names
        #     }
         
        bankdata={"name":name,"account_number":account_number,"ifsc":ifsc,"micr":micr} 
        bankdata=json.dumps(bankdata)
        resdata={}
        resdata=json.dumps(resdata)
        cursor.execute("INSERT INTO ocrdetails(company_name,request_data, source,requested_for,response_data,status,request_id,company_id) VALUES (%s, %s, %s, %s,%s,%s,%s,%s)",(company, bankdata,sources,"adhaar",resdata,"Completed",req_id,token))
        
        
        conn.commit()
        
        response_data = {
        "status": "fail",
        "message": "Processed successfully",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data":"" ,
        "error" :str(e) ,
        "req_id":req_id,
        "sources" : sources  
                
        }
        logger.info(f"Error_response_data_extract_bank_details",exc_info=True,extra={"response_data":response_data})
        
        # log_entry = {
        # "request": request_data,
        # "response": response_data
        # }
        traceback.print_exc()
        # savelogs(log_entry,log_file_path)
        print(f"Error: {e}")
        return jsonify({"msg": f"An error occurred while processing the request: {str(e)}","success":False}), 500



def get_template(company,documentid):
    static_payload = {"username": "akankshak@bpwealth.com", "password":"APItest123"}
    print(f"static {static_payload}")

    url='https://pdf.meon.co.in/editor/token/'
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, headers=headers, json=static_payload)

    print(response.json())
    token=response.json()
    token=token["access"]


    url = f"https://pdf.meon.co.in/editor/get_template/{documentid}/"

    payload = {}
    headers = {
    'Accept': 'application/json, text/plain, */*',
    'Authorization': f'Bearer {token}',
    'Connection': 'keep-alive',
    'Referer': 'https://pdf.meon.co.in/'
    
    }

    response = requests.request("POST", url, headers=headers, data=payload)


    data1=response.json()


    tempeles=data1["data"]
    return  tempeles

@profile
@app.route('/extract_pdf_data_byurl', methods=['POST'])
def readpdf_data():
    try:
        token="19"
        # conn, cursor = connectDB()
        data = request.get_json()
        document =data['document']
        felds_data=data["fields"]
        req_id = data.get('req_id')
        company = data.get('company')
        sources = data.get('sources')
        path=f"static/pdfs/{sources}/{company}/"
        documentid=data["document_id"]
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        response = requests.get(document, stream=True)
        if response.status_code != 200:
            return {"success": False, "msg": "Failed to download file"}
        
        file_content = BytesIO(response.content)  # Read file into memory
        file_content.seek(0)  # Reset pointer
        original_extension = os.path.splitext(document.split("/")[-1].split("?")[0])[-1].lower()
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" in content_type or document.lower().endswith(".pdf"):
            original_extension=".pdf"
        doc=documentid.split("-")[0]+documentid.split("-")[1]
        save_path = os.path.join(path, f"{req_id}{doc}{original_extension}")
        
        with open(save_path, "wb") as f:
                f.write(file_content.getbuffer())
                
        if not save_path or not os.path.exists(save_path):
            return jsonify({'msg': 'error in download pdf'}), 404
                
        
        template= get_template(company,documentid)
        ocrdata={}
        matching_results={}
        total=0
        felds=felds_data.keys()
        print(felds,"felds")
        # for  i in template:
        #     # print("yesi",documentid,i.get("document_id"))
            
        #     if documentid==i.get("document_id"):
        lenfelds=len(felds)
        matchlen=0
        document_template=template.get("document_template")
        with pdfplumber.open(save_path) as pdf:
            for j in document_template.items():
                        # print("yesj")
                        
                        for z in felds:
                            # print("yesz")
                            
                            if z in j[1].keys():
                                if not ocrdata.get(z) :
                                            codinates=j[1].get(z)
                                            # page = doc[int(j[0])-1]'
                                    
                                    
                                            page = pdf.pages[int(j[0])-1]
                                            page_height = page.height
                                            padding = 5
                                            h=codinates[0]["height"]
                                            w=codinates[0]["width"]
                                            x_coordinate=codinates[0]["x-coordinate"]
                                            y_coordinate=codinates[0]["y-coordinate"]
                                            # print(page,z,codinates)
                                            # exractdata = fitz.Rect(x_coordinate, y_coordinate, x_coordinate + w, y_coordinate + h)
                                            # exractdata = page.get_text("text", clip=exractdata)
                                            
                                            clipped_text = page.within_bbox((x_coordinate, y_coordinate-10, x_coordinate +w+7, y_coordinate + h-15)).extract_text()
                                            if not clipped_text:
                                                print("andr aa gya ",clipped_text)
                                                clipped_text = page.within_bbox((x_coordinate, y_coordinate-10, x_coordinate +w+50, y_coordinate + h-5)).extract_text()
                                                
                                            print(clipped_text)
                                            if z not in matching_results.keys():
                                                totalenty=True
                                            else:
                                                totalenty=False
                                                
                                            if not ocrdata.get(z):
                                                ocrdata[z]=clipped_text
                                            matching_results[z]=string_matching_percentage(ocrdata.get(z).replace(" ", "").lower(), felds_data.get(z).replace(" ", "").lower())
                                            if totalenty and  ocrdata.get(z):  
                                                total=total+int(matching_results[z])
                                                print(total,"total")
                                                matchlen=matchlen+1
                                                
                                                
                                            if matchlen== lenfelds:
                                                break 
                                        
                                        

                                    
                                 
                                
        
        
        average_match_percentage=total/len(felds)       
        return jsonify({"matching_results":matching_results,"extracted_data":ocrdata,"average_match_percentage":average_match_percentage,"success":True}), 200              
    except Exception as e:
                traceback.print_exc()
        



def get_template1(companyid,documentid):
    conn, cursor = connectDB()
    try:
        
        cursor.execute("select pdf_username, pdf_password from companydetails where company_id=%s ",(companyid,))
        data1 =cursor.fetchall()[0]
        static_payload={"username":data1[0],"password":data1[0]}
    except:
        static_payload = {"username": "democapital@Meon.co.in", "password":"democapital"}
    print(f"static {static_payload}")

    url='https://pdf.meon.co.in/editor/token/'
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, headers=headers, json=static_payload)

    print(response.json())
    token=response.json()
    token=token["access"]


    url = f"https://pdf.meon.co.in/editor/get_template/{documentid}/"

    payload = {}
    headers = {
    'Accept': 'application/json, text/plain, */*',
    'Authorization': f'Bearer {token}',
    'Connection': 'keep-alive',
    'Referer': 'https://pdf.meon.co.in/'
    
    }

    response = requests.request("POST", url, headers=headers, data=payload)


    data1=response.json()


    tempeles=data1["data"]
    return  tempeles


@profile
@jwt_required() 
@app.route('/extract_pdf_details_byurl', methods=['POST'])
def extract_pdf_details_byurl():

        # conn, cursor = connectDB()
        token=get_jwt_identity()
        data = request.get_json()
        document =data['document']
        felds_data=data["fields"]
        req_id = data.get('req_id')
        company = data.get('company')
        sources = data.get('sources')
        path=f"static/pdfs/{sources}/{company}/"
        documentid=data["document_id"]
        if not os.path.exists(path):
            os.makedirs(path)
        
        response = requests.get(document, stream=True)
        if response.status_code != 200:
            return {"success": False, "msg": "Failed to download file"}
        
        file_content = BytesIO(response.content)  # Read file into memory
        file_content.seek(0)  # Reset pointer
        original_extension = os.path.splitext(document.split("/")[-1].split("?")[0])[-1].lower()
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" in content_type or document.lower().endswith(".pdf"):
            original_extension=".pdf"
        doc=documentid.split("-")[0]+documentid.split("-")[1]
        save_path = os.path.join(path, f"{req_id}{doc}{original_extension}")
        
        with open(save_path, "wb") as f:
                f.write(file_content.getbuffer())
                
        if not save_path or not os.path.exists(save_path):
            return jsonify({'msg': 'error in download pdf'}), 404
                
        
        template= get_template1(token,documentid)
        ocrdata={}
        matching_results={}
        total=0
        felds=felds_data.keys()
        print(felds,"felds")
        # for  i in template:
        #     # print("yesi",documentid,i.get("document_id"))
            
        #     if documentid==i.get("document_id"):
        lenfelds=len(felds)
        matchlen=0
        document_template=template.get("document_template")
        # with pdfplumber.open(save_path) as pdf:
        doc = fitz.open(save_path)
        for j in document_template.items():
                        # print("yesj")
                        
                        for z in felds:
                            # print("yesz")
                            
                            if z in j[1].keys():
                                if not ocrdata.get(z) :
                                    codinates=j[1].get(z)
                                    # page = doc[int(j[0])-1]'
                                    try:
                                        
                                            page = doc[int(j[0])-1]
                                            # page_height = page.height
                                            padding = 5
                                            h=codinates[0]["height"]
                                            w=codinates[0]["width"]
                                            x_coordinate=codinates[0]["x-coordinate"]
                                            y_coordinate=codinates[0]["y-coordinate"]
                                            # print(page,z,codinates)
                                            # exractdata = fitz.Rect(x_coordinate, y_coordinate, x_coordinate + w, y_coordinate + h)
                                            # exractdata = page.get_text("text", clip=exractdata)
                                            
                                            # clipped_text = page.within_bbox((x_coordinate, y_coordinate-10, x_coordinate +w+7, y_coordinate + h-15)).extract_text()
                                            # if not clipped_text:
                                            #     print("andr aa gya ",clipped_text)
                                            #     clipped_text = page.within_bbox((x_coordinate, y_coordinate-10, x_coordinate +w+50, y_coordinate + h-5)).extract_text()
                                            rect = fitz.Rect(x_coordinate, y_coordinate-10 , x_coordinate + w , y_coordinate + h-10 )
                                            clipped_text = page.get_text("text", clip=rect).strip()

                                            # Fallback if empty
                                            if not clipped_text:
                                                fallback_rect = fitz.Rect(x_coordinate, y_coordinate - 10, x_coordinate + w + 50, y_coordinate + h - 5)
                                                clipped_text = page.get_text("text", clip=fallback_rect).strip()

                                            
                                                
                                            print(clipped_text)
                                            if z not in matching_results.keys():
                                                totalenty=True
                                            else:
                                                totalenty=False
                                                
                                            if not ocrdata.get(z):
                                                ocrdata[z]=clipped_text
                                            matching_results[z]=string_matching_percentage(ocrdata.get(z).replace(" ", "").lower(), felds_data.get(z).replace(" ", "").lower())
                                            if totalenty and  ocrdata.get(z):  
                                                total=total+int(matching_results[z])
                                                matchlen=matchlen+1
                                                print(total,"total")
                                                
                                            
                                        
                                            if matchlen== lenfelds:
                                                break

                                        
                                    
                                    except Exception as e:
                                        traceback.print_exc()
        
        
        average_match_percentage=total/len(felds)       
        return jsonify({"matching_results":matching_results,"extracted_data":ocrdata,"average_match_percentage":average_match_percentage,"success":True}), 200              

# ------------------------------------------------------super admin--------------------------------------------------------
@app.route('/addcompany', methods=['POST'])
def addcompany():
    data = request.get_json()
    try:
    
        company=data["company"]
        
        company_id=data["company_id"]
        account_type=data["account_type"]
        
        pan_access=data.get("pan_access")
        aadhar_access=data.get("aadhar_access")
        financial_proof_access=data.get("financial_proof_access")
        bank_proof_access=data.get("bank_proof_access")
        pdf_access=data.get("pdf_access")
        
        if pan_access==True:
             pan_access=1
        else:
             pan_access=0  
             
                 
        if aadhar_access==True:
            aadhar_access=1
        else:
            aadhar_access=0
            
            
        if financial_proof_access==True:
            financial_proof_access=1
        else:
            financial_proof_access=0
            
        
        if bank_proof_access==True:
            bank_proof_access=1
        else:
            bank_proof_access=0
        if pdf_access==True:
            pdf_access=1
        else:
            pdf_access=0
            
            
                
        
        
        
        credit=data.get("credit")
        
        if not credit:
            credit=0
        credit=int(credit)    
        
        
            
        
        negative_credit_allowed=data["negative_credit_allowed"]

        email=data["email"]
        password=data["password"]
        active_status=1
        
        
        if negative_credit_allowed:
            negative_credit_allowed=1
        else:
            negative_credit_allowed=0
            
            
        if not negative_credit_allowed:
            negative_credit_allowed=1
            
            
        conn, cursor = connectDB()
        
        
        # company=data["company"]
        
        cursor.execute("INSERT INTO companydetails(company_name, email, password, company_id, account_type, active_status, total_credits,credits_left,is_negative_credit_allowed,pan_access,aadhar_access,financial_proof_access,bank_proof_access,pdf_access) VALUES (%s, %s, %s, %s, %s, %s, %s, %s,%s,%s,%s,%s,%s,%s)",(company, email, password, company_id, account_type,active_status, credit,credit,negative_credit_allowed,pan_access,aadhar_access,financial_proof_access,bank_proof_access,pdf_access))
        
        
        conn.commit()
        
        return jsonify({"msg":f"company added sucessfully ","success":True})
    
    except Exception as e  :
        traceback.print_exc()
        return jsonify({"msg":f"{e}","success":False})
        
        
        
    
    
    
    
    
    
    
    
    
@app.route('/activate_deactivate_company', methods=['POST'])
def activate_deactivate_company():
    data = request.get_json()
    active=data["active"]
    company=data["company"]
    conn, cursor = connectDB()
    
    try:
        if active==True:
            
            cursor.execute(" update companydetails set  active_status =%s where company_name=%s ",(1,company))
            conn.commit()
            return jsonify({"msg":f"company activated ","success":True})
            
        elif active==False:
            
            cursor.execute(" update companydetails set  active_status =%s where  company_name=%s ",(0,company))
            conn.commit()
            
            return jsonify({"msg":f"company deactivated ","success":True})
    except Exception as e :
        traceback.print_exc()
        return jsonify({"msg":f"{e}","success":False})
        
            
        
        
    




@app.route('/add_credit', methods=['POST'])
def add_credit():
    
    data = request.get_json()
    try:
        credit=data["credit"]
        company=data["company"]
        conn, cursor = connectDB()
        cursor.execute(" select total_credits ,credits_left  from companydetails where  company_name=%s ",(company,))
        data1 =cursor.fetchall()
        total_credits=data1[0][0]
        credits_left=data1[0][1]
        total_credits=int(credit)+int(credits_left)
        credits_left=int(credit)+int(credits_left)
        
        
        
        cursor.execute(" update companydetails set  total_credits =%s ,credits_left=%s where  company_name=%s ",(total_credits,credits_left,company))
        conn.commit()
        return jsonify({"msg":f"company credit added  ","success":True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"msg":f"{e}","success":False})
        



@app.route('/remove_credit', methods=['POST'])
def remove_credit():
    data = request.get_json()
    try:
        credit=data["credit"]
        company=data["company"]
        conn, cursor = connectDB()
        cursor.execute(" select total_credits ,credits_left  from companydetails where  company_name=%s ",(company,))
        data1 =cursor.fetchall()
        total_credits=data1[0][0]
        credits_left=data1[0][1]
        total_credits=int(credits_left)-int(credit)
        credits_left=int(credits_left)-int(credit)
        
        
        
        cursor.execute(" update companydetails set  total_credits =%s ,credits_left=%s where  company_name=%s ",(total_credits,credits_left,company))
        conn.commit()
        return jsonify({"msg":f"credit removed   ","success":True}) 
    except Exception as e:
        traceback.print_exc()
        return jsonify({"msg":f"{e}","success":False})


# @app.route('/add_access/<company>', methods=['POST'])
# def add_credit():
#     pass


@app.route('/add_document_access', methods=['POST'])
@jwt_required() 
def add_access():
    data = request.get_json()
    token=get_jwt_identity()
    try:
        pan_access=data.get("pan_access")
        aadhar_access=data.get("aadhar_access")
        financial_proof_access=data.get("financial_proof_access")
        bank_proof_access=data.get("bank_proof_access")
        pdf_access=data.get("pdf_access")
        
        # company=data["company"]
        conn, cursor = connectDB()
        
        if pan_access==True:
                cursor.execute(" update companydetails set  pan_access =%s where  company_id=%s ",(1,token))
                conn.commit()
                
            
            
        if aadhar_access==True:
                cursor.execute(" update companydetails set  aadhar_access =%s where  company_id=%s ",(1,token))
                conn.commit()
                
        if financial_proof_access==True:
                cursor.execute(" update companydetails set  financial_proof_access =%s where  company_id=%s ",(1,token))
                conn.commit()
                
        if bank_proof_access==True:
                cursor.execute(" update companydetails set  bank_proof_access =%s where  company_id=%s ",(1,token))
                conn.commit()   
        if pdf_access ==True  :
            cursor.execute(" update companydetails set  pdf_access =%s where  company_id=%s ",(1,token))
            conn.commit()
                
                
                
        
          
          
            
        # cursor.execute(" update companydetails set  document_access =%s where  company_id=%s ",(0,token))
        # conn.commit()
        return jsonify({"msg":f"access granted   ","success":True}) 
    except Exception as e:
        traceback.print_exc()
        return jsonify({"msg":f"{e}","success":False})


@app.route('/remove_document_access', methods=['POST'])
@jwt_required()
def remove_access():
    data = request.get_json()
    token=get_jwt_identity()
    try:
        # removedocument_access=data.get("document_access")
        pan_access=data.get("pan_access")
        aadhar_access=data.get("aadhar_access")
        financial_proof_access=data.get("financial_proof_access")
        bank_proof_access=data.get("bank_proof_access")
        pdf_access=data.get("pdf_access")
        
        # removedocument_access=removedocument_access.split(",")
        # company=data["company"]
        conn, cursor = connectDB()
    
        if pan_access==True:
                cursor.execute(" update companydetails set  pan_access =%s where  company_id=%s ",(0,token))
                conn.commit()
                
            
            
        if aadhar_access==True:
                cursor.execute(" update companydetails set  aadhar_access =%s where  company_id=%s ",(0,token))
                conn.commit()
                
        if financial_proof_access==True:
                cursor.execute(" update companydetails set  financial_proof_access =%s where  company_id=%s ",(0,token))
                conn.commit()
                
        if bank_proof_access==True:
                cursor.execute(" update companydetails set  bank_proof_access =%s where  company_id=%s ",(0,token))
                conn.commit()   
                
        if pdf_access ==True  :
            cursor.execute(" update companydetails set  pdf_access =%s where  company_id=%s ",(0,token))
            conn.commit()     
        # cursor.execute(" update companydetails set  document_access =%s where  company_name=%s ",(document_access,company))
        # conn.commit()
        return jsonify({"msg":f"access removed   ","success":True}) 
    except Exception as e:
        traceback.print_exc()
        return jsonify({"msg":f"{e}","success":False})







@app.route('/get_document_access', methods=['POST'])
@jwt_required()
def get_access():
    # data = request.get_json()
    token=get_jwt_identity()
    
    try:
        conn, cursor = connectDB()
        cursor.execute("select aadhar_access, financial_proof_access, bank_proof_access,pan_access,pdf_access from companydetails where company_id=%s ",(token,))
        data1 =cursor.fetchall()[0]
        if not data1:
            return jsonify({"msg":" comoany not registered","success":False})
            
        # cursor.execute(" update companydetails set  document_access =%s where  company_name=%s ",(document_access,company))
        # conn.commit()
        
        data={}
        if data1[0]=="1" or data1[0]==1:
            data["aadhar_access"]=True
        else:
            data["aadhar_access"]=False
            
        if data1[1]=="1" or data1[1]==1:
            data["financial_proof_access"]=True
        else:
            data["financial_proof_access"]=False
            
        if data1[2]=="1" or data1[2]==1:
            data["bank_proof_access"]=True
        else:
            data["bank_proof_access"]=False
            
        if data1[3]=="1" or data1[3]==1:
            data["pan_access"]=True
        else:
            data["pan_access"]=False
            
        if data1[4]=="1" or data1[4]==1:
            data["pdf_access"]=True
        else:
            data["pdf_access"]=False
        
        
        
            
        return jsonify({"data":data,"success":True}) 
    except Exception as e:
        traceback.print_exc()
        return jsonify({"msg":f"{e}","success":False})





@app.route('/enable_disable_negative_credit', methods=['POST'])
def enable_disable_negative_credit():
    data = request.get_json()
    try:
        negative_credit_allowed=data["negative_credit_allowed"]
        company=data["company"]
        conn, cursor = connectDB()
    
        if negative_credit_allowed==True:
            
            cursor.execute(" update companydetails set  is_negative_credit_allowed =%s where company_name=%s ",(1,company))
            conn.commit()
            return jsonify({"msg":f"negative_credit activated ","success":True})
            
        elif negative_credit_allowed==False:
            
            cursor.execute(" update companydetails set  is_negative_credit_allowed =%s where  company_name=%s ",(0,company))
            conn.commit()
            
            return jsonify({"msg":f"negative_credit deactivated ","success":True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"msg":f"{e}","success":False})



@app.route('/change_account_type', methods=['POST'])
def change_account_type():
    data = request.get_json()
    try:
        account_type=data["account_type"]
        company=data["company"]
        conn, cursor = connectDB()
        
        cursor.execute(" update companydetails set  account_type =%s where  company_name=%s ",(account_type,company))
        conn.commit()
        
        return jsonify({"msg":f"Account type changed to  {account_type}  ","success":True}) 
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"msg":f"{e}","success":False})
    








@app.route('/add_pdf_credentials', methods=['POST'])
def add_pdf_credentials():
    data = request.get_json()
    try:
        pdf_username=data["pdf_username"]
        pdf_password=data["pdf_password"]
        company=data["company"]
        conn, cursor = connectDB()
        
        cursor.execute(" update companydetails set  pdf_username=%s ,  pdf_password=%s  where company_name=%s  ",(pdf_username,pdf_password,company))
        conn.commit()
        
        return jsonify({"msg":f"credentials added  ","success":True}) 
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"msg":f"{e}","success":False})

import requests
#from paddleocr import PaddleOCR
import easyocr

#app = Flask(__name__)

paddle_ocr = PaddleOCR(lang='en')
easy_reader = easyocr.Reader(['en'])

BANK_IFSC_PREFIXES = [
    "SBIN", "ICIC", "HDFC", "PUNB", "UTIB", "IBKL", "BARB", "KARB",
    "YESB", "CNRB", "IOBA", "ORBC", "MAHB", "INDB", "SCBL", "HSBC","BDBL","KKBK"
]

def clean_text(text):
    return text.upper().replace('\n', ' ').replace('|', 'I').replace('.', '').replace(':', '')

def normalize_ifsc(code):

    code = code.upper().replace('O', '0').replace('D','0').replace('|', 'I')
    code = code.replace('\u200b', '').replace(':', '').strip()

    for prefix in BANK_IFSC_PREFIXES:
        idx = code.find(prefix)
        if idx != -1:
            # Slice from prefix to the next space or end of string
            sliced = code[idx:].split()[0]  # gets till first space
            sliced = re.sub(r'\W', '', sliced)  # strip non-alphanum just in case
            match = re.search(r'[A-Z]{4}0[A-Z0-9]{6}', sliced)
            if match:
                return match.group()

    # Fallback: try general IFSC match
    match = re.search(r'[A-Z]{4}0[A-Z0-9]{6}', code)
    if match:
        return match.group()

    return code

def validate_relaxed_ifsc(code):

    if code.startswith("IFSC"):
        return False 
    return bool(re.fullmatch(r'[A-Z]{4}0[A-Z0-9]{6}', code))

def extract_ifsc_from_prefix(text, bank_prefixes=BANK_IFSC_PREFIXES):
 
    cleaned_text = text.upper().replace('O', '0').replace('|', 'I').replace('\u200B', '')
    print("Trying prefix based searching for IFSC....")

    for prefix in bank_prefixes:
        idx = cleaned_text.find(prefix)
        if idx != -1:
            candidate = cleaned_text[idx:idx + 11]
            if re.fullmatch(r'[A-Z]{4}0[A-Z0-9]{6}', candidate):
                print(f"✅ IFSC found via prefix fallback: {candidate}")
                return candidate
    return None



def extract_ifsc_code_from_text(text):
    text = clean_text(text)
    candidates = re.findall(r'[A-Z0-9]{6,13}', text)
    valid_ifscs = []
    for raw in candidates:
        norm = normalize_ifsc(raw)
        print("🔍 Candidate:", norm)
        if validate_relaxed_ifsc(norm):
            valid_ifscs.append(norm)

    ifsc_lines = re.findall(r'IFS(?:C|0|O)?[\s\-:=]*[Cc]ode[\s\-:=]*([A-Z0-9\s]{10,20})', text)

    for raw in ifsc_lines:
        parts = re.split(r'\s+', raw)
        for part in parts:
            norm = normalize_ifsc(part)
            if re.fullmatch(r'[A-Z]{4}0[A-Z0-9]{6}', norm):
                print("IFSC from IFSC line segment:", norm)
                return norm

    keyword_matches = re.findall(r'IFS[C0O][\s\-:=]*([A-Z0-9]{6,13})', text)

    for match in keyword_matches:
        norm = normalize_ifsc(match)
        if len(norm) <= 8 and norm.startswith('0'):
            for prefix in BANK_IFSC_PREFIXES:
                patched = prefix + norm
                if validate_relaxed_ifsc(patched):
                    print(f" Fixed IFSC near keyword: {patched}")
                    return patched
        if validate_relaxed_ifsc(norm):
            print(f"IFSC near keyword: {norm}")
            return norm

    # Prefer known bank prefixes
    for prefix in BANK_IFSC_PREFIXES:
        for code in valid_ifscs:
            if code.startswith(prefix):
                print(f"IFSC by prefix: {code}")
                return code

    prefix_ifsc = extract_ifsc_from_prefix(text)
    if prefix_ifsc:
        return prefix_ifsc

    print("Still no valid IFSC")
    return None


def extract_account_number_easyocr(image):
    try:
        results = easy_reader.readtext(image)
        text = ' '.join([res[1] for res in results])
        numbers = re.findall(r'\b\d{9,18}\b', text)
        if numbers:
            acc = max(numbers, key=len)
            print("🎯 Account number from EasyOCR:", acc)
            return acc
    except Exception as e:
        print("EasyOCR account number extraction failed:", e)
    return None

@app.route('/api/extract-cheque-info', methods=['POST'])
def extract_cheque_info():
    #token= "19"
    #sta, msg = check_auth(token, "cheque_ocr")
    #if not sta:
     #   return jsonify({"msg": msg, "success": False})
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}),400
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        try:
            results = paddle_ocr.predict(img)
        except Exception as e:
            return jsonify({"error": f"PaddleOCR crashed: {str(e)}"}), 500

        text_blocks = results[0].get("rec_texts", []) if results else []
        full_text = ftfy.fix_text(' '.join(text_blocks))

        #conn, cursor = connectDB()
        #cursor.execute("SELECT credits_left FROM companydetails WHERE company_id=%s", (token,))
        #data1 = cursor.fetchall()
        #credits_left = int(data1[0][0]) - 1

        #cursor.execute("UPDATE companydetails SET credits_left=%s WHERE company_id=%s", (credits_left, token))
        #conn.commit()

        print("📝 OCR TEXT:\n", full_text)
        ifsc_code = extract_ifsc_code_from_text(full_text)
        account_number = extract_account_number_easyocr(img)
        #conn, cursor = connectDB()
        #cursor.execute("SELECT credits_left FROM companydetails WHERE company_id=%s", (token,))
        #data1 = cursor.fetchall()
        #credits_left = int(data1[0][0]) - 1
        #cursor.execute("UPDATE companydetails SET credits_left=%s WHERE company_id=%s", (credits_left, token))
        #conn.commit()

        return jsonify({
            "ifsc_code": ifsc_code,
            "account_number": account_number,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    

import math
import base64

#app = Flask(__name__)


def is_straight_line_signature(image, width_expand=4.0, row_ratio_thresh=0.95, debug_path="cropped.png", pixel_threshold=9.5):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh_clean = cv2.subtract(thresh, vertical_lines)
    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if not contours:
        print(" No contours found.")
        return False

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    center, (w, h), angle = rect
    print(f" Rotated Box — Center: ({int(center[0])}, {int(center[1])}), W: {w:.2f}, H: {h:.2f}, Angle: {angle:.2f}")

    if w < h:
        w *= width_expand
    else:
        h *= width_expand

    rect = (center, (w, h), angle)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    size = (int(w), int(h))
    try:
        roi = cv2.getRectSubPix(rotated, size, center)
    except Exception as e:
        print(f"Could not extract ROI: {e}")
        return False

    if roi.size == 0:
        print(" Empty ROI.")
        return False
    
    _, clean_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    vertical_sum = np.sum(clean_bin == 255, axis=1)
    non_zero_rows = np.count_nonzero(vertical_sum > 0)
    vertical_ratio = non_zero_rows / roi.shape[0]
    print(f" Vertical spread: {non_zero_rows} / {roi.shape[0]} = {vertical_ratio:.2f}")

    top_rows = roi[:5, :]
    bottom_rows = roi[-5:, :]
    touches_top = np.any(np.sum(top_rows == 255, axis=1) > pixel_threshold)
    touches_bottom = np.any(np.sum(bottom_rows == 255, axis=1) > pixel_threshold)
    print(f"touches_top: {touches_top}, touches_bottom: {touches_bottom}")

    debug = image.copy()
    cv2.drawContours(debug, [box], 0, (0, 255, 0), 2)
    cv2.drawContours(debug, [cnt], 0, (255, 0, 0), 1)
    cv2.imwrite(debug_path, debug)
    print(f"Saved debug: {debug_path}")

    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
    print(f"Aspect Ratio: {aspect_ratio:.2f}")

    if (
        (vertical_ratio > 0.97 and not touches_top and not touches_bottom and aspect_ratio > 2.5) or
        (vertical_ratio < 0.3 and aspect_ratio > 10 and not touches_top and not touches_bottom)
    )   :
        print("Final verdict: Straight line signature detected.")
        return True
    else:
        print(" Final verdict: Normal signature.")
        return False
    
def is_background_white(image, mean_thresh=110, stddev_thresh=80, color_diff_thresh=35, edge_density_thresh=0.09, margin=10):
    h, w, _ = image.shape

    top = image[0:margin, :, :]
    bottom = image[-margin:, :, :]
    left = image[:, 0:margin, :]
    right = image[:, -margin:, :]
    
    border_pixels = np.vstack([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ])

    mean_val = np.mean(border_pixels)
    std_val = np.std(border_pixels)
    color_diff = np.max(border_pixels, axis=1) - np.min(border_pixels, axis=1)
    max_color_diff = np.max(color_diff)

    combined_border = np.concatenate([
        top,
        bottom,
        cv2.resize(left, (top.shape[1], margin), interpolation=cv2.INTER_AREA),
        cv2.resize(right, (top.shape[1], margin), interpolation=cv2.INTER_AREA)
    ], axis=0)

    gray_border = cv2.cvtColor(combined_border, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_border, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Logs
    print(f" Border Mean: {mean_val:.2f}, StdDev: {std_val:.2f}, Max RGB Diff: {max_color_diff}")
    print(f" Edge Density in Border: {edge_density:.4f}")

    return (
        mean_val > mean_thresh and
        std_val < stddev_thresh and
        max_color_diff < color_diff_thresh and
        edge_density < edge_density_thresh
    )
def get_signature_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(" No contours found.")
        return 0
    cnt = max(contours, key=cv2.contourArea)
    data_pts = cnt.reshape(-1, 2).astype(np.float32)
    mean = np.mean(data_pts, axis=0)
    centered = data_pts - mean
    _, eigenvectors = cv2.PCACompute(centered, mean=None)
    angle = math.degrees(math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]))
    print(f" Signature Orientation Angle: {angle:.2f}°")
    return angle

#1. take in file 2. check if mage or pdf, if pdf convert to image 3. check if background white, if no return unacceptable background color , yes go check orientation, if already horizontal, then 
# return what we were returning earlier, is_horizontal = true and is_background_white = true. if not horizontal, then rotate and return the base64 of rotated image.. 4. initially make a success_key.
#initialise it with true. at any sort of error, like file not uploaded, or backgrounf not white or rotation needed , make success key false, this lets us know if the uploaded check needs to be updated in any way
def is_signature_horizontal(image, angle_threshold=40):
    angle = get_signature_angle(image)
    normalized_angle = abs(angle % 180)
    return normalized_angle < angle_threshold or abs(normalized_angle - 180) < angle_threshold

def rotate_image_to_horizontal(image):

    h, w = image.shape[:2]
    if w >= h:
        print("Already horizontal.")
        return image
    
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return rotated

def image_to_base64(img):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG")
    base64_str = base64.b64encode(buffer.getvalue()).decode()
    return base64_str

@app.route('/api/verify-signature-format', methods=['POST'])
def verify_signature_format():
    #token = "19"
    #sta, msg = check_auth(token, "sign_verification")
    #if not sta:
     #  return jsonify({"msg": msg, "success": False})
    try:
        success = True  

        file = request.files.get('file')
        if not file:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        filename = file.filename.lower()
        temp_path = "/tmp/sign_image.jpg"

        ALLOWED_EXTENSIONS = ['.pdf', '.jpg', '.png']
        if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
             print("User did not upload an image or pdf")
             return jsonify({
                  "success": False,
                    "error": "Invalid file format. Acceptable formats: .PDF, .JPG, .PNG"
         }), 400


        if filename.endswith('.pdf'):
            print("PDF Uploaded. Converting to Image.")
            file.stream.seek(0)
            pdf_success = convert_pdf_to_image(file, temp_path)
            if not pdf_success:
                return jsonify({"success": False, "error": "PDF conversion failed"}), 500
            img = cv2.imread(temp_path)
        else:
            img_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is not None and img.shape[-1] == 4:
            print("Image has alpha channel — converting transparent background to white.")
            alpha_channel = img[:, :, 3]
            rgb_channels = img[:, :, :3]
            white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255
            alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
            img = (rgb_channels * alpha_factor + white_background * (1 - alpha_factor)).astype(np.uint8)


        if img is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400
           
        #conn,cursor = connectDB()
        #cursor.execute("SELECT credits_left FROM companydetails WHERE company_id=%s", (token,))
        #data1 = cursor.fetchall()
        #credits_left = int(data1[0][0]) - 1
        #cursor.execute("UPDATE companydetails SET credits_left=%s WHERE company_id=%s", (credits_left, token))
        #conn.commit()
        # Check for invalid straight-line signature
        if is_straight_line_signature(img):
            print("Invalid signature — only a straight line detected")
            return jsonify({
                "success": False,
                "error": "Invalid signature"
                      }), 200

        background_ok = is_background_white(img)
        if not background_ok:
            print("Background not in correct format")
            return jsonify({
                "success": False,
                "error": "Unacceptable image background",
                "is_background_white": False
            }), 200

        horizontal = is_signature_horizontal(img)
        
        if horizontal:
            return jsonify({
                "success": True,
                "is_horizontal": True,
                "is_background_white": True
            }), 200
        else:
            rotated = rotate_image_to_horizontal(img)
            base64_img = image_to_base64(rotated)
            return jsonify({
                "success": False,
                "is_horizontal": False,
                "is_background_white": True,
                "rotated_image": base64_img
            }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/verify-signature-format_byurl', methods=['POST'])
def verify_signature_format_byurl():
   #token = "19"
   #sta, msg = check_auth(token, "sign_verification")
   #if not sta:
    #   return jsonify({"msg": msg, "success": False})

    try:

        file_url = request.form.get('file_url') or (request.get_json() or {}).get('file_url')
        if not file_url:
            return jsonify({"success": False, "error": "No file URL provided"}), 400

        response = requests.get(file_url, stream=True)
        if response.status_code != 200:
            return jsonify({"success": False, "error": "Failed to download file from URL"}), 400

        content_type = response.headers.get("Content-Type", "").lower()
        file_content = BytesIO(response.content)
        file_content.seek(0)

        raw_filename = file_url.split("/")[-1].split("?")[0].lower()
        if '.' in raw_filename:
            extension = os.path.splitext(raw_filename)[-1]
        elif "pdf" in content_type:
            extension = ".pdf"
        elif "jpeg" in content_type or "jpg" in content_type:
            extension = ".jpg"
        elif "png" in content_type:
            extension = ".png"
        else:
            extension = ""

        if extension not in ['.pdf', '.jpg', '.png']:
            return jsonify({
                "success": False,
                "error": "Invalid file format. Acceptable formats: .PDF, .JPG, .PNG"
            }), 400

        filename = f"downloaded{extension}"
        temp_path = "/tmp/sign_image_byurl.jpg"

        if extension == ".pdf":
            file = FileStorage(stream=file_content, filename=filename, content_type="application/pdf")
            pdf_success = convert_pdf_to_image(file, temp_path)
            if not pdf_success:
                return jsonify({"success": False, "error": "PDF conversion failed"}), 500
            img = cv2.imread(temp_path)
        else:
            img_arr = np.frombuffer(file_content.read(), np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[-1] == 4:
            print("Image has alpha channel — converting transparent background to white.")
            alpha_channel = img[:, :, 3]
            rgb_channels = img[:, :, :3]
            white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255
            alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
            img = (rgb_channels * alpha_factor + white_background * (1 - alpha_factor)).astype(np.uint8)

        if img is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400

        # Credit deduction
     #  conn, cursor = connectDB()
      # cursor.execute("SELECT credits_left FROM companydetails WHERE company_id=%s", (token,))
      # data1 = cursor.fetchall()
      # credits_left = int(data1[0][0]) - 1
      #cursor.execute("UPDATE companydetails SET credits_left=%s WHERE company_id=%s", (credits_left, token))
      # conn.commit()
        if is_straight_line_signature(img):
            print("Invalid signature — only a straight line detected")
            return jsonify({
                "success": False,
                "error": "Invalid signature"
                      }), 200

        background_ok = is_background_white(img)
        if not background_ok:
            return jsonify({
                "success": False,
                "error": "Unacceptable image background",
                "is_background_white": False
            }), 200

        horizontal = is_signature_horizontal(img)

        if horizontal:
            return jsonify({
                "success": True,                                                                                                                                               
                "is_horizontal": True,
                "is_background_white": True
            }), 200
        else:
            rotated = rotate_image_to_horizontal(img)
            base64_img = image_to_base64(rotated)
            return jsonify({
                "success": True,
                "is_horizontal": False,
                "is_background_white": True,
                "rotated_image": base64_img
            }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


from datetime import datetime

app = Flask(__name__)

paddle_ocr = PaddleOCR(use_textline_orientation=True,lang='en')

def detect_osd_angle(image):
    try:
        config = "--psm 0"
        osd = pytesseract.image_to_osd(image, config=config)
        for line in osd.split('\n'):
            if "Rotate" in line:
                angle = int(line.split(":")[-1].strip())
                print(f"[detect_osd_angle] Detected angle: {angle}")
                return angle
    except pytesseract.TesseractError as e:
        print("[detect_osd_angle] OSD failed:", e)
    except Exception as e:
        print("[detect_osd_angle] Unexpected error:", e)
    print("[detect_osd_angle] Defaulting to angle = 0")
    return 0  # safe fallback


def rotate_image(img, angle):
 
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(img, rot_mat, (new_w, new_h), borderValue=(255, 255, 255))


def correct_rotation(img, angle):
    if angle == 0:
        return img
    print("correcting orientation")
    return rotate_image(img, -angle) 


def run_ocr(img):
    result = paddle_ocr.predict(img)
    lines = result[0].get("rec_texts", [])
    return '\n'.join(lines)

def extract_name_from_text(text: str) -> str:
    lines = text.upper().split('\n')

    for i, line in enumerate(lines):
        if "NAME" in line:
            match = re.search(r'NAME[:\-]?\s*([A-Z ]{3,})', line)
            if match:
                return match.group(1).strip()

            for j in range(i + 1, min(i + 3, len(lines))):
                candidate = lines[j].strip()
                if re.fullmatch(r'[A-Z ]{3,}', candidate) and not any(x in candidate for x in ["S/W/D", "DOB", "ADDRESS"]):
                    return candidate

    return None


def extract_address(text: str) -> list:
    text_upper = text.upper()
    lines = text_upper.split('\n')
    address_lines = []
    collect = False

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        if re.match(r'^(ADDRESS|ADD)\b|^(ADDRESS|ADD)[A-Z]', line_stripped):
            collect = True
            cleaned_line = re.sub(r'^(ADDRESS|ADD)[:\-\s]*', '', line_stripped)
            if cleaned_line:
                address_lines.append(cleaned_line)
            continue

        if collect:
            if any(stop_word in line_stripped for stop_word in [
                "PIN", "AUTH", "DATE", "VALID", "ISSUING", "FORM", "CLASS", "SIGNATURE", "COV"
            ]):
                break

            if line_stripped:
                address_lines.append(line_stripped)

            if len(address_lines) >= 2:
                break

    return address_lines
def address_breaker(address, l, index):
            # print(address.strip()[-1])
            if address.strip()[-1] == "0":
                address = address.strip()[:-1]
            splitedaddress = address.replace("  ", " ").split(" ")
            add1 = ""
            add2 = ""
            add3 = ""
            add1full = False
            add2full = False
            add3full = False
            for i in splitedaddress:
                if len(add1) + len(i) < l and add1full == False:
                    add1 += i + " "
                else:
                    add1 = add1.strip()
                    add1full = True
                if len(add2) + len(i) < l and add1full == True and add2full == False:
                    add2 += i + " "
                elif len(add2) + len(i) >= l:
                    add2 = add2.strip()
                    add2full = True
                if len(add3) + len(i) < l and add2full == True:
                    add3 += i + " "
                elif len(add3) + len(i) >= l and add2full == True:
                    add3 = add3.strip()
                    break
            if index == 1:
                return str(add1.strip())
            if index == 2:
                return str(add2.strip())
            if index == 3:
                return str(add3.strip())

def extract_dl_number(text: str) -> str:
    text = ftfy.fix_text(text)
    text_upper = text.upper()
    lines = text_upper.split('\n')

    normalized_text = text_upper.replace(" ", "").replace("-", "")

    pattern = r'\b[A-Z]{2}\d{2}[A-Z]?\d{4,5}\d{5,8}\b'
    match = re.search(pattern, normalized_text)
    if match:
        return match.group()

    for i, line in enumerate(lines):
        if 'DL' in line or 'LICENCE NO' in line or 'LICENCE' in line:
            for j in range(i, min(i+2, len(lines))):  # Check same + next line
                cleaned = lines[j].replace(" ", "").replace("-", "")
                match = re.search(pattern, cleaned)
                if match:
                    return match.group()

    return None

def extract_pincode(address_lines, full_text):
    def normalize_digits(text):
        return (
            text.upper()
                .replace('O', '0')
                #.replace('I', '1')
                .replace('L', '1')
                .replace('|', '1')
        )

    for line in reversed(address_lines):
        norm_line = normalize_digits(line)
        print("Checking address line:", norm_line)
        match = re.search(r'(?<!\d)(\d{6})(?!\d)', norm_line) 
        if match:
            print("PIN found in address line:", match.group(1))
            return match.group(1)

    norm_text = normalize_digits(full_text)
    match = re.search(r'PIN[:\s\-]*?(\d{6})\b', norm_text)
    if match:
        print(" PIN found in raw text:", match.group(1))
        return match.group(1)

    print(" Still no PIN found.")
    return None


def extract_dl_info(text: str) -> dict:
    raw_text = text
    text = ftfy.fix_text(text)

    name = extract_name_from_text(raw_text)
    
    dl_number = extract_dl_number(raw_text)

    address_lines = extract_address(raw_text)
    merged_address = ' '.join(address_lines).strip()
    add1 = address_breaker(merged_address,20,1)
    add2 = address_breaker(merged_address,20,2)
    add3 = address_breaker(merged_address,20,3)

    structured_address = {}
    if add1: structured_address["line_1"] = add1
    if add2: structured_address["line_2"] = add2
    if add3: structured_address["line_3"] = add3


    pincode = extract_pincode(address_lines, raw_text)

    date_matches = re.findall(r'\d{2}[-/]\d{2}[-/]\d{4}', text)
    
    def parse_flexible_date(d):
        d = d.replace('/', '-')
        return datetime.strptime(d, '%d-%m-%Y')

    try:
        unique_dates = sorted(set(date_matches), key=parse_flexible_date)
    except Exception:
        unique_dates = date_matches

    dob = unique_dates[0] if unique_dates else None

    return {
        "dl_number": dl_number,
        "name": name,
        "address": structured_address,
        "pincode": pincode,
        "date_of_birth": dob
    }

@app.route('/api/extract-dl-info', methods=['POST'])
def extract_dl_info_api():
    try:
        token = "19" 
        sta, msg = check_auth(token, "dl_access")
        if not sta:
            return jsonify({"msg": msg, "success": False}), 403

        file = request.files.get("file")
        req_id = request.form.get("req_id")
        sources = request.form.get("sources")
        dl_number = request.form.get("dl_number")

        if not sources:
            return {"msg": "Please provide sources", "success": False}
        if not file:
            return {"msg": "No file uploaded", "success": False}

        if not sources:
            return {"msg": "Please provide sources", "success": False}

        path = f"static/dlfiles/{sources}"
        os.makedirs(path, exist_ok=True)

        filename = f"{dl_number or 'dl_unknown'}.jpg"
        save_path = os.path.join(path, filename)

        file.save(save_path)  
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        angle = detect_osd_angle(img)
        corrected_img = correct_rotation(img, angle)
        result = paddle_ocr.predict(corrected_img)
        text_lines = result[0].get("rec_texts", [])
        raw_text = '\n'.join(text_lines)
        print("OCR TEXT:\n", raw_text)

        extracted = extract_dl_info(raw_text)
        extracted["raw_text"] = raw_text
        extracted["req_id"] = req_id

        return jsonify(extracted)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    

#passport api

paddle = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    cpu_threads=4,
    det_model_dir='models/ch_PP-OCRv3_det_infer',
    rec_model_dir='models/ch_PP-OCRv3_rec_infer',
    cls_model_dir='models/ch_ppocr_mobile_v2.0_cls_infer'
)


def enhance_contrast_and_sharpen(img):
    print("[enhance_contrast_and_sharpen] Enhancing contrast and sharpness")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray)

    contrast_bgr = cv2.cvtColor(contrast_img, cv2.COLOR_GRAY2BGR)

    blurred = cv2.GaussianBlur(contrast_bgr, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(contrast_bgr, 2.0, blurred, -1.0, 0)


    return sharpened


def preprocess_for_osd(image):

    h, w = image.shape[:2]
    if h < 800:
        image = cv2.resize(image, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(image)

    return contrast_img


def upscale_image(img, scale=2.0):
    print(f"[upscale_image] Upscaling image by {scale}x using INTER_CUBIC")
    height, width = img.shape[:2]
    return cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)

def extract_passport_number(text):
    print("[extract_passport_number] scanning text...")
    match = re.search(r'\b([A-Z][0-9]{7})\b', text)
    if match:
        print("[extract_passport_number] found:", match.group(1))
    return match.group(1) if match else None

def extract_surname(lines, mrz_lines):
    print("[extract_surname] checking lines for label...")
    forbidden_keywords = ['GIVEN', 'NAMES', 'NAME']

    for i, line in enumerate(lines):
        line_upper = line.upper()
        # Fuzzy match: allow SURNAME, SUMAME, SURMAME, etc.
        if re.search(r'\bS[UO]R?N?A?M[E]?\b', line_upper):
            cleaned_inline = re.sub(r'.*S[UO]R?N?A?M[E]?\b[:\-\s/]*', '', line_upper).strip()
            if cleaned_inline and re.fullmatch(r'[A-Z]{2,}', cleaned_inline):
                print("[extract_surname] found inline:", cleaned_inline)
                return cleaned_inline

            # Check next 2 lines if current one is noisy
            for j in range(1, 3):
                if i + j < len(lines):
                    candidate = lines[i + j].strip().upper()
                    if (
                        re.fullmatch(r'[A-Z]{2,}', candidate)
                        and not any(kw in candidate for kw in forbidden_keywords)
                    ):
                        print(f"[extract_surname] found safe fallback in line {i + j}:", candidate)
                        return candidate

    # MRZ fallback (last resort)
    if mrz_lines:
        part = mrz_lines[0].split('<<')[0]  # Only use first MRZ line
        surname = part.replace('P<IND', '').strip('<')
        print("[extract_surname] found via MRZ:", surname)
        return surname

    print("[extract_surname] not found")
    return None
def extract_given_name(lines, mrz_lines):
    print("[extract_given_name] checking lines for label...")

    for i, line in enumerate(lines):
        line_clean = line.strip().upper()

        # Fuzzy match for label: catch variants like GIVENNAMELS, GIVEN NAME(S), etc.
        if 'GIVEN' in line_clean and 'NAME' in line_clean:
            print(f"[extract_given_name] fuzzy label match on line {i}: {line_clean}")

            # Try extracting inline value
            value = re.sub(r'.*GIVEN\s*NAME[S]?[):\/\-\s]*', '', line_clean).strip()
            value = re.sub(r'[^A-Z ]+', '', value)

            # If inline value is too short or just noise, try next line
            if len(value) < 3 or 'GIVEN' in value:
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip().upper()
                    if re.fullmatch(r'[A-Z ]{2,}', next_line):
                        print(f"[extract_given_name] using next line {i+1}:", next_line)
                        return next_line
            else:
                print("[extract_given_name] found via inline cleanup:", value)
                return value

    # Fallback to MRZ
    if mrz_lines:
        parts = mrz_lines[0].split('<<')
        if len(parts) >= 2:
            given = parts[1].replace('<', ' ').strip()
            print("[extract_given_name] fallback from MRZ:", given)
            return given

    print("[extract_given_name] not found")
    return None

#df extract_sex(lines):
#   print("[extract_sex] searching lines...")
 #  for i, line in enumerate(lines):
#       if 'SEX' in line:
 #          print("[extract_sex] found 'SEX' label at line", i)
  #         for j in range(1, 4):
   #            if i + j < len(lines):
    #               possible = lines[i + j].strip()
     #              if re.fullmatch(r'[MF]', possible):
      #                 print("[extract_sex] found value:", possible)
       #                return possible
        #   m = re.search(r'\bSEX[:\-]?\s*(F|M)\b', line)
         #  if m:
          #     print("[extract_sex] found inline value:", m.group(1))
           #    return m.group(1)
   #print("[extract_sex] not found")
   #return None

def extract_dates(text):
    print("[extract_dates] finding all date-like patterns...")
    dates = re.findall(r'\d{2}[-/]\d{2}[-/]\d{4}', text)
    try:
        dates = sorted(set(dates), key=lambda d: datetime.strptime(d.replace('/', '-'), '%d-%m-%Y'))
    except:
        print("[extract_dates] date parsing error")
    print("[extract_dates] extracted dates:", dates)
    dob = dates[0] if len(dates) > 0 else None
    doi = dates[1] if len(dates) > 1 else None
    doe = dates[2] if len(dates) > 2 else None
    return dob, doi, doe

def extract_passport_info(text: str) -> dict:
    print("[extract_passport_info] cleaning and splitting text")
    lines = ftfy.fix_text(text.upper()).split('\n')
    full_text = '\n'.join(lines)
    mrz_lines = [line for line in lines if '<<' in line]

    data = {
        "passport_number": extract_passport_number(full_text),
        "surname": extract_surname(lines, mrz_lines),
        "given_name": extract_given_name(lines, mrz_lines),
       #"sex": extract_sex(lines),
    }

    dob, doi, doe = extract_dates(full_text)
    data["date_of_birth"] = dob
    data["date_of_issue"] = doi
    data["date_of_expiry"] = doe

    return data


def extract_address_and_pin(text: str):
    lines = [line.strip() for line in text.upper().split('\n') if line.strip()]
    address_lines = []
    collecting = False

    # Primary method: Find label with ADDRESS or ADD
    for i, line in enumerate(lines):
        if not collecting and re.search(r'\b(ADDRESS|ADD)\b', line):
            print(f"[extract_address_and_pin] Found address label in line {i}: {line}")
            collecting = True
            for j in range(1, 4):
                if i + j < len(lines):
                    address_lines.append(lines[i + j].strip())
            break

    # Fallback method: Use PIN line and 2 above it
    if not collecting:
        print("[extract_address_and_pin] No address label found. Trying fallback using PIN.")
        for i, line in enumerate(lines):
            fixed_line = line.replace("O", "0") 
            if re.search(r'\b(\d{6})\b', fixed_line):
                print(f"[extract_address_and_pin] Found PIN-like pattern in line {i}: {line}")
                start = max(i - 2, 0)
                address_lines = lines[start:i + 1]
                collecting = True
                break

    if not collecting:
        print("[extract_address_and_pin] No address found at all.")
        return [], None

    pin_code = None
    for line in address_lines:
        fixed_line = line.replace("O", "0")
        match = re.search(r'\b(\d{6})\b', fixed_line)
        if match:
            pin_code = match.group(1)
            break

    print(f"[extract_address_and_pin] Extracted lines:\n{address_lines}")
    print(f"[extract_address_and_pin] Extracted pin: {pin_code}")
    return address_lines, pin_code

def extract_family_members(lines):
    print("[extract_family_members] scanning for family fields...")
    data = {
        "father_or_guardian": None,
        "mother": None,
        "spouse": None
    }

    def find_name(start_idx):
        for j in range(1, 3):  
            if start_idx + j < len(lines):
                candidate = lines[start_idx + j].strip().upper()
                if re.fullmatch(r"[A-Z\s]{3,}", candidate):
                    return candidate
        return None

    for i, line in enumerate(lines):
        l = re.sub(r'[^A-Z]', '', line.upper())  


        # Father or Guardian
        if any(k in l for k in ["FATHER", "GUARDIAN", "LEGALGUARDIAN"]):
            print(f"[extract_family_members] Triggered on father/guardian at line {i}: {line}")
            name = find_name(i)
            if name:
                data["father_or_guardian"] = name

        # Mother
        if re.search(r'\bMOTHE?R?\b', line.upper()):
            print(f"[extract_family_members] Triggered on mother at line {i}: {line}")
            name = find_name(i)
            if name:
                data["mother"] = name

        # Spouse
        clean_line = re.sub(r'[^A-Z]', '', line.upper())
        if any(kw in clean_line for kw in ['SPOUSE', 'SPOUS']):
            print(f"[extract_family_members] Triggered on spouse at line {i}: {line}")
            name = find_name(i)
            if name:
                data["spouse"] = name


    return data


@app.route('/api/extract-passport-info' , methods = ['POST'])
def extract_passport_combined():
    try:
       # token = "19"
        #sta,msg = check_auth(token,"passport_access")
       # if not sta:
       #     return jsonify({"msg":msg , "success": False}), 403
        
        file_front = request.files.get('file_front')
        file_rear = request.files.get('file_rear')
       # req_id = request.form.get("req_id")
        #sources = request.form.get("sources")
       # passport_number = request.form.get("passport_number")


        if not file_front or not file_rear:
            return jsonify({"success": False, "error": "Both file_front and file_rear are required"}), 400
        #if not sources:
            #return {"msg": "Please provide sources", "success": False}
        
        #path = f"static/passports/{sources}"
        #os.makedirs(path , exist_ok = True)

        #filename_front = f"{passport_number or req_id}front.jpg"
        #filename_rear = f"{passport_number or req_id}rear.jpg"
        #save_path_front = os.path.join(path,filename_front)
        #save_path_rear = os.path.join(path,filename_rear)

        #file_front.save(save_path_front)
        #file_rear.save(save_path_rear)

        def process_image(file):
            filename = file.filename.lower()
            temp_path = "/tmp/temp_img.jpg"

            if filename.endswith('.pdf'):
                file.stream.seek(0)
                if not convert_pdf_to_image(file, temp_path):
                    raise ValueError("PDF conversion failed")
                img = cv2.imread(temp_path)
            else:
                img_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            return img

        # Process front
        img_front = process_image(file_front)
        angle = 0
        try:
            angle = detect_osd_angle(img_front)
        except pytesseract.TesseractError:
            img_front = preprocess_for_osd(img_front)
            try:
                angle = detect_osd_angle(img_front)
            except pytesseract.TesseractError:
                pass

        img_front = correct_rotation(img_front, angle)
        img_front = upscale_image(img_front, 2.0)
        img_front = enhance_contrast_and_sharpen(img_front)
        results = paddle.ocr(img_front, cls=True)
        text_front = '\n'.join([line[1][0] for box in results for line in box])

        print("Page 1: \n")
        print("RAW OCR:", text_front)
        passport_info = extract_passport_info(text_front)

        # Process rear
        img_rear = process_image(file_rear)
        img_rear = upscale_image(img_rear, 2.0)
        img_rear = enhance_contrast_and_sharpen(img_rear)
        results = paddle.ocr(img_rear, cls=True)
        text_rear = '\n'.join([line[1][0] for box in results for line in box])

        print("Page 2: \n")
        print("RAW OCR: " ,text_rear)
        address, pin = extract_address_and_pin(text_rear)
        lines_rear = text_rear.splitlines()
        family_info = extract_family_members(lines_rear)

        return jsonify({
            "success": True,
            "passport_info": passport_info,
            "address_info": {
                "address": address,
                "pin": pin
            },
            "family_info": family_info
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    

# masking adhar num api

def mask_aadhaar_number(img, ocr_result):
    print("[mask_aadhaar_number] Scanning OCR results...")
    aadhaar_regex = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")

    matches = 0

    for line in ocr_result[0]:
        box, (text, conf) = line
        cleaned_text = text.replace("O", "0").replace("I", "1").replace("|", "1")

        match = aadhaar_regex.search(cleaned_text)
        if match:
            aadhaar_raw = match.group(0)
            aadhaar_digits = re.sub(r"\D", "", aadhaar_raw)

            if len(aadhaar_digits) == 12:
                print(f"[match] Aadhaar detected: {aadhaar_raw} at approx {tuple(map(int, box[0]))}")

                box_arr = np.array(box).astype(int)
                x_min = min(pt[0] for pt in box_arr)
                x_max = max(pt[0] for pt in box_arr)
                y_min = min(pt[1] for pt in box_arr)
                y_max = max(pt[1] for pt in box_arr)

                width = x_max - x_min
                height = y_max - y_min

                print(f"[box] width={width}, height={height}")

                if height > 100 or height < 10:
                    print("[skip] Bounding box height too extreme, skipping...")
                    continue

                h, w = img.shape[:2]
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)

                print(f"[masking] Aadhaar digits: {aadhaar_digits}, Mask width: FULL ({width})")

                char_width = (x_max - x_min) // 12
                mask_end_x = x_min + char_width * 8

                cv2.rectangle(img, (x_min, y_min), (mask_end_x, y_max), (0, 0, 0), thickness=-1)

                matches += 1
                if matches >= 2:
                    break  
    return img



@app.route("/api/mask-aadhaar", methods=["POST"])
def mask_aadhaar_api():
    file = request.files.get("file")
    if not file:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    filename = file.filename.lower()
    temp_path = "/tmp/aadhaar_input.jpg"

    if filename.endswith(".pdf"):
        file.stream.seek(0)
        success = convert_pdf_to_image(file, temp_path)
        if not success:
            return jsonify({"success": False, "error": "PDF conversion failed"}), 500
        img = cv2.imread(temp_path)
    else:
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"success": False, "error": "Invalid image"}), 400

    original_img = img.copy()  # For raw OCR
    preprocessed_img = img.copy()

    # Rotation handling
    try:
        angle = detect_osd_angle(preprocessed_img)
    except pytesseract.TesseractError:
        preprocessed_img = preprocess_for_osd(preprocessed_img)
        angle = detect_osd_angle(preprocessed_img)

    preprocessed_img = correct_rotation(preprocessed_img, angle)
    original_img = correct_rotation(original_img, angle)

    preprocessed_img = upscale_image(preprocessed_img, scale=2.0)
    preprocessed_img = enhance_contrast_and_sharpen(preprocessed_img)

    # OCR on both versions
    ocr_raw = paddle.ocr(original_img)
    ocr_pre = paddle.ocr(preprocessed_img)

    # Merge both results
    combined_ocr = [ocr_raw[0] + ocr_pre[0]]

    # Mask Aadhaar numbers on original image using merged OCR
    print("\n--- OCR RAW ---")
    for line in ocr_raw[0]:
        print(line[1][0])

    print("\n--- OCR PREPROCESSED ---")
    for line in ocr_pre[0]:
        print(line[1][0])

    masked_img = mask_aadhaar_number(original_img, combined_ocr)
    base64_img = image_to_base64(masked_img)

    return jsonify({
        "success": True,
        "masked_image_base64": base64_img
    })


@app.route("/api/mask-aadhaar-url", methods=["POST"])
def mask_aadhaar_by_url_api():
    file_url = request.form.get("file_url") or (request.get_json() or {}).get("file_url")
    if not file_url:
        return jsonify({"success": False, "error": "No file URL provided"}), 400

    temp_path = "/tmp/aadhaar_input.jpg"

    try:
        response = requests.get(file_url, stream=True)
        if response.status_code != 200:
            return jsonify({"success": False, "error": "Failed to download file from URL"}), 400

        content_type = response.headers.get("Content-Type", "").lower()
        file_bytes = BytesIO(response.content)

        # Handle PDF or image
        if "pdf" in content_type or file_url.lower().endswith(".pdf"):
            file_bytes.seek(0)
            success = convert_pdf_to_image(file_bytes, temp_path)
            if not success:
                return jsonify({"success": False, "error": "PDF conversion failed"}), 500
            img = cv2.imread(temp_path)
        else:
            file_bytes.seek(0)
            img_array = np.asarray(bytearray(file_bytes.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    if img is None:
        return jsonify({"success": False, "error": "Invalid image"}), 400

    original_img = img.copy()
    preprocessed_img = img.copy()

    try:
        angle = detect_osd_angle(preprocessed_img)
    except pytesseract.TesseractError:
        preprocessed_img = preprocess_for_osd(preprocessed_img)
        angle = detect_osd_angle(preprocessed_img)

    preprocessed_img = correct_rotation(preprocessed_img, angle)
    preprocessed_img = upscale_image(preprocessed_img, scale=2.0)
    preprocessed_img = enhance_contrast_and_sharpen(preprocessed_img)

    ocr_raw = paddle.ocr(original_img)
    ocr_pre = paddle.ocr(preprocessed_img)

    combined_ocr = [ocr_raw[0] + ocr_pre[0]]

    print("\n--- OCR RAW ---")
    for line in ocr_raw[0]:
        print(line[1][0])

    print("\n--- OCR PREPROCESSED ---")
    for line in ocr_pre[0]:
        print(line[1][0])

    masked_img = mask_aadhaar_number(original_img, combined_ocr)
    base64_img = image_to_base64(masked_img)

    return jsonify({
        "success": True,
        "masked_image_base64": base64_img
    })



# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5008, debug=True)
if (__name__ == "__main__"):
    print('om chal gya ')
#     # socketio.run(app, host = "0.0.0.0", port=5107)
    # http = WSGIServer(('0.0.0.0',5008), app)
    # http.serve_forever()
    app.run(host="0.0.0.0", port=5008)




