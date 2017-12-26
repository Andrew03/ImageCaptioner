import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders
import argparse

def send_email(args):
  msg = MIMEMultipart()
  msg['From'] = args.user
  msg['To'] = args.to
  msg['Subject'] = args.subject

  for file_name in args.txt_files:
    f = file(file_name)
    attachment = MIMEText(f.read())
    attachment.add_header('Content-Disposition', 'attachment', filename=file_name)
    f.close()
    msg.attach(attachment)

  for file_name in args.png_files:
    f = file(file_name)
    attachment = MIMEImage(f.read())
    f.close()
    msg.attach(attachment)

  if not args.body:
    body = 'No desired body'
    content = MIMEText(body, 'plain')
    msg.attach(content)
  else:
    f = open(args.body)
    body = MIMEText(f.read())
    f.close()
    msg.attach(body)

  server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
  server.ehlo()
  server.login(args.user, args.password)
  server.sendmail(args.user, args.to, msg.as_string())
  server.close()
  print 'Email sent!'

def main(args):
  send_email(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--user', type=str, required=True,
                      help='Email address of the user')
  parser.add_argument('--password', type=str, required=True,
                      help='Password for email account of user')
  parser.add_argument('--to', type=str, required=True,
                      help='Email address of the receiver')
  parser.add_argument('--subject', type=str,
                      default='Email sent from the terminal',
                      help='Subject of the email. Default value of \'Email send from the terminal\'')
  parser.add_argument('--body', type=str,
                      default='',
                      help='File containing body of the message to send to the receiver')
  parser.add_argument('--attach_png', action='append', dest='png_files',
                      default=[],
                      help='png file to be sent to the receiver.')
  parser.add_argument('--attach_txt', action='append', dest='txt_files',
                      default=[],
                      help='txt file to be sent to the receiver.')
  args = parser.parse_args()
  main(args)
