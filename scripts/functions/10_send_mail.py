# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 09:45:22 2021

@author: Alexandre.Iborra
"""

# Renvoie la prediction pour une image de la vidéo
def send_email(img_score):
    
    # Lien vers les images pour le mail
    path_img = chemin_appli + "/data/demonstration/dechet_detect/predicted_img.png"
    path_check = chemin_appli + "/data/demonstration/img/checked.png"
    path_uncheck = chemin_appli + "/data/demonstration/img/unchecked.png"
    
    # Variable pour indiquer le lieux du déchet (a remplacer avec information sur l'image)
    var_lieux = "Boulevard Barrieu a Royat"
    sujet = "Un déchet a été détecté !"
    
    # Envoie du mail
    print("Envoi du mail en cours...")
    msg = EmailMessage()
    
    # set the plain text body
    msg.set_content('This is a plain text body.')
    
    #msg.set_payload(body,"utf8")
    # On définit l'entête du mail
    msg['Subject'] = sujet
    msg['From'] = formataddr((sender_name, sender_email))
    msg['To'] = formataddr((receiver_names, receiver_emails))

    # Création d'un content-ID pour les images
    image_cid = make_msgid(domain='xyz.com')
    image_check = make_msgid(domain='xyz.com')
    image_uncheck = make_msgid(domain='xyz.com')

    # Définition du HTML pour le mail
    msg.add_alternative("""\
    <html>
    <body>
    <p>Bonjour,<br>
       Nous avons d&eacute;tect&eacute; un potentiel d&eacute;p&ocirc;t sauvage via le dispositif exp&eacute;rimental situ&eacute; {lieux}.<br>
       Sur une &eacute;chelle de 0 &agrave; 100 la volum&eacute;trie identifi&eacute;e de ce d&eacute;p&ocirc;t sauvage a un score de {score_depot}.
    </p>
    <center><img src="cid:{image_cid}" height=300"" width="400"></center>
    <p> Pourriez-vous confirmer qu&#039;il s&#039;agit bien d&#039;un d&eacute;p&ocirc;t sauvage sur l&#039;image ci-dessous en cliquant sur l&#039;un des liens ? </p>
    <center>
        <a href="https://fr.wikipedia.org/wiki/Test", style="color:#32CD32;">
		    <img src="cid:{image_check}", width = "20px"> 
		    Oui, il s&#039;agit bien d&#039;un d&eacute;p&ocirc;t sauvage 
	        </a>
    </center>
    <p><center> Pensez &agrave; envoyer les moyens adapt&eacute;s pour son enl&egrave;vement </center></p>
    <br>
    <center>
	    <a href="https://fr.wikipedia.org/wiki/Test", style="color:#FF0000;">
		<img src="cid:{image_uncheck}", width = "20px">
		Non, il ne s&#039;agit pas d&#039;un d&eacute;p&ocirc;t sauvage
	    </a>
    </center>
    <p><center> L&#039;image sera sauvegard&eacute;e pour am&eacute;liorer les prochaines pr&eacute;dictions </center></p>
    <br>
    <p> En vous remerciant pour votre aide, nous vous souhaitons une agr&eacute;able journ&eacute;e. </p>

    </body>
    </html>
    """.format(image_cid=image_cid[1:-1],image_check=image_check[1:-1],image_uncheck=image_uncheck[1:-1],lieux=var_lieux, score_depot=img_score), subtype='html')
   
    # now open the image and attach it to the email
    with open(path_img, 'rb') as img, open(path_check, 'rb') as check, open(path_uncheck, 'rb') as uncheck:
        # know the Content-Type of the image
        maintype, subtype = mimetypes.guess_type(img.name)[0].split('/')
        maintype_c, subtype_c = mimetypes.guess_type(check.name)[0].split('/')
        maintype_u, subtype_u = mimetypes.guess_type(uncheck.name)[0].split('/')

        # attach it
        msg.get_payload()[1].add_related(img.read(), 
                                             maintype=maintype, 
                                             subtype=subtype, 
                                             cid=image_cid)
    
        msg.get_payload()[1].add_related(check.read(), 
                                         maintype=maintype_c, 
                                         subtype=subtype_c, 
                                         cid=image_check)
    
        msg.get_payload()[1].add_related(uncheck.read(), 
                                         maintype=maintype_u, 
                                         subtype=subtype_u, 
                                         cid=image_uncheck)
    try:
        # Encrypts the email
        context = ssl.create_default_context()
        server = smtplib.SMTP_SSL(smtp_adress, smtp_port, context = context)
        server.connect(smtp_adress, smtp_port)
        #with smtplib.SMTP_SSL(smtp_adress, smtp_port, context = context) as server:
        # connexion au compte
        server.login(sender_email, password)
          # envoi du mail
        
        text = msg.as_string().encode('ascii')

        server.sendmail(sender_email, receiver_emails, text)
        print('Email sent!')
    finally:
        print('Closing the server...')
        server.quit()
        
