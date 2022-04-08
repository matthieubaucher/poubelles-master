# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:10:24 2021

@author: 33675
"""

# Dossier du modèle qui permet de faire la prédiction
chdir(chemin_appli + "/model")

# Récupération de l'image à envoyer
filename = glob.glob("../data/demonstration/dechet_detect/*")
filename = "../data/demonstration/dechet_detect/img_test.jpeg"

# Configuration du html
email_html = open('../data/demonstration/email.html')
email_body = email_html.read()

for receiver_email, receiver_name in zip(receiver_emails, receiver_names):
        print("Sending the email...")
        # Configurating user's info
        msg = MIMEMultipart()
        msg['To'] = formataddr((receiver_name, receiver_email))
        msg['From'] = formataddr((sender_name, sender_email))
        msg['Subject'] = 'Hello, my friend ' + receiver_name
        
        msg.attach(MIMEText(email_body, 'html'))

        try:
            # Open PDF file in binary mode
            with open(filename, "rb") as attachment:
                            part = MIMEBase("application", "octet-stream")
                            part.set_payload(attachment.read())

            # Encode file in ASCII characters to send by email
            encoders.encode_base64(part)

            # Add header as key/value pair to attachment part
            part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {filename}",
            )

            msg.attach(part)
        except Exception as e:
                print(f'Oh no! We didn\'t found the attachment!\n{e}')
                break

        try:
                # Creating a SMTP session | use 587 with TLS, 465 SSL and 25
                #server = smtplib.SMTP('smtp.gmail.com', 587)
                # Encrypts the email
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(smtp_adress, smtp_port, context = context) as server:
                    # connexion au compte
                    server.login(sender_email, password)
                      # envoi du mail
                    text = msg.as_string()
                    server.sendmail(sender_email, receiver_emails, text)
                #server.starttls(context=context)
                # We log in into our Google account
                #server.login(sender_email, password)
                # Sending email from sender, to receiver with the email body
                #server.sendmail(sender_email, receiver_email, msg.as_string())
                    print('Email sent!')
        except Exception as e:
            print(f'Oh no! Something bad happened!\n{e}')
            break
        finally:
            print('Closing the server...')
            server.quit()
