# Assimilation de données : BLUE et filtre de Kalman d’ensemble

Bureau d’études ISAE-SUPAERO combinant analyse statistique et assimilation séquentielle. Les scripts étudient l’estimation d’un état à partir d’une ébauche et d’observations bruitées, puis un filtre de Kalman d’ensemble (EnKF) sur le modèle de Lorenz-95.

## Organisation

- `BE_DA_LMS/Statistical_Analysis/statistical_analysis.py` : analyse BLUE et diagnostics.
- `BE_DA_LMS/EnsembleMethods/EnKF_to_complete.py` : expérience jumelle, cycles EnKF et erreur RMSE.
- `BE_DA_LMS/EnsembleMethods/models.py` : modèle de dynamique.
- `BE_DA_LMS/EnsembleMethods/localization.py` : localisation des covariances.
- [HOUSSENALY_DACHER_BE_DA.pdf](HOUSSENALY_DACHER_BE_DA.pdf) : rapport.
- `plots/` : figures enregistrées ; `BE-Sujet-DA-2026_SXS.pdf` : énoncé.

## Explorer les expériences

Avec Python 3, NumPy, SciPy et Matplotlib :

```sh
python -m pip install numpy scipy matplotlib
python BE_DA_LMS/Statistical_Analysis/statistical_analysis.py
python BE_DA_LMS/EnsembleMethods/EnKF_to_complete.py
```

Les paramètres de simulation se règlent dans les scripts. Le fichier EnKF inclut notamment la taille d’ensemble, les options d’inflation et de localisation et une graine aléatoire. Il affiche des diagnostics et des graphiques selon les options activées.

Ces points d’entrée ont été vérifiés par lecture du code, sans nouvelle exécution ; les versions des dépendances ne sont pas verrouillées.

## Origine du code

Les scripts s’appuient sur du matériel pédagogique du **CERFACS**. Leurs en-têtes mentionnent notamment Anthony Fillion, Mayeul Destouches, Eliott Lumet et Selime Gurol, ainsi que la licence **CeCILL-C**. Ces crédits et dispositions sont à conserver ; le dépôt ne revendique pas la création originale de l’ensemble du code.
