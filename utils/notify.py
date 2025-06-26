import os
import smtplib
from email.message import EmailMessage
from twilio.rest import Client

# Twilio config
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
NOTIFICATION_PHONE_NUMBER = os.environ.get('NOTIFICATION_PHONE_NUMBER')

# Email config
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
EMAIL_ADDRESS = os.environ.get('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')
EMAIL_RECEIVER = os.environ.get('EMAIL_RECEIVER')

def send_bin_full_alert(bin_name):
    message = f"🚨 Alert: The {bin_name} bin is full. Please empty it as soon as possible."

    # Send SMS via Twilio
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER and NOTIFICATION_PHONE_NUMBER:
        try:
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            msg = client.messages.create(
                body=message,
                from_=TWILIO_PHONE_NUMBER,
                to=NOTIFICATION_PHONE_NUMBER
            )
            print(f"SMS sent: SID {msg.sid}")
        except Exception as e:
            print(f"Error sending SMS: {e}")
    else:
        print("Twilio environment variables missing.")

    # Send Email via SMTP
    if EMAIL_ADDRESS and EMAIL_PASSWORD and EMAIL_RECEIVER:
        try:
            msg = EmailMessage()
            msg.set_content(message)
            msg['Subject'] = f"Bin Alert: {bin_name} is full"
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = EMAIL_RECEIVER

            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as smtp:
                smtp.starttls()
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                smtp.send_message(msg)
                print("Email sent successfully.")
        except Exception as e:
            print(f"Error sending email: {e}")
    else:
        print("Email environment variables missing.")
