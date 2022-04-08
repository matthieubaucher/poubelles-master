# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:12:29 2021

@author: 33675
"""

# Chargement de l'environnement
print("Chargement de l'environnement...")
# chemin_appli = "C:/Users/Alexandre.Iborra/Desktop/Veolia/application"
chemin_appli = "C:/Users/33675/Documents/Professionel/Talan/Projets/Veolia/application"

exec(open(chemin_appli + "/scripts/00_init_env.py", encoding="utf-8").read())

exec(open(chemin_appli + "/scripts/01_load_models.py", encoding="utf-8").read())
