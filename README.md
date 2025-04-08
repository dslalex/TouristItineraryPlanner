# 🧭 Planification d’itinéraires touristiques personnalisés

---

## 🧩 Description :
Pour un touriste visitant une ville pendant un temps limité, on souhaite planifier automatiquement un itinéraire quotidien incluant un sous-ensemble de points d’intérêts (musées, monuments…) qui maximisent son plaisir. C’est le Tourist Trip Design Problem (TTDP), qui est une variante de problème d’orientation (Orienteering Problem) avec des contraintes supplémentaires propres au tourism (Branch-and-Check Approaches for the Tourist Trip Design Problem with Rich Constraints | Request PDF)】. On dispose d’une liste de lieux, chacun avec une durée de visite et un intérêt (score), parfois des horaires d’ouverture, et on doit choisir lesquels visiter et dans quel ordre, de façon à ne pas dépasser la durée totale de la journée (ou la durée de séjour), et en respectant les plages horaires de chaque lieu. D’autres contraintes peuvent s’ajouter : visites obligatoires (le touriste a absolument voulu X), catégories à limiter (pas plus de 2 musées dans la journée), ordre imposé (déjeuner après telle visite), etc (Branch-and-Check Approaches for the Tourist Trip Design Problem with Rich Constraints | Request PDF)】. L’objectif est souvent d’optimiser la somme des intérêts des lieux visités (ou la satisfaction du touriste) sous ces contraintes.

---

## 🔍 Intérêt de l’approche CSP :
Ce problème combine une sélection combinatoire (sous-ensemble de lieux) et un ordonnancement (ordre de visite avec temps de trajet entre lieux). Une approche par contraintes permet d’intégrer aisément tous les types de contraintes mentionnées (faisabilité temporelle, fenêtres d’ouverture, quotas de catégories, obligations) dans un seul modèle. Par exemple, on peut utiliser des variables booléennes pour indiquer si un lieu est visité, des contraintes de temps cumulatif pour s’assurer que le parcours entre 9h et 18h rentre dans la durée disponible, et des contraintes de chemin pour la logique de parcours. Duc Minh Vu et al. ont proposé une approche exacte (branch-and-check) où le problème maître sélectionne les lieux et un problème esclave vérifie la possibilité d’un circuit temporel – c’est une approche hybride CP/PLNE qui utilise des contraintes pour valider les contraintes riches (ordre, types (Branch-and-Check Approaches for the Tourist Trip Design Problem with Rich Constraints | Request PDF)】. L’approche CSP pure pourrait consister à créer une séquence de positions horaires (1er lieu, 2ème lieu, etc.) avec des contraintes de transition (si lieu A en position 1 et B en position 2, alors respecter temps trajet + durée(A) ≤ heure(B)), etc. La force du CSP est de pouvoir pruner l’exploration en éliminant tôt les séquences impossibles (par ex. si ajouter tel lieu rend le total > temps dispo, ce branchement est coupé). De plus, on peut optimiser la somme des scores via une technique de branch and bound sur le CSP. Bien que NP-difficile, des solveurs peuvent trouver des itinéraires optimaux pour un nombre raisonnable de POIs, et fournir des options alternatives en fonction des préférences. Cela dépasse largement ce qu’un simple algorithme glouton ferait, et garantit de ne pas rater une combinaison de lieux peu évidente mais très avantageuse.

---

## 📁 Structure du projet

- **[`data`](data)**: Contient les informations des graphes de villes
  - 🏙️ city_graph.py: Implémentation générique du graphe de ville
  - 🗼 paris_graph.py: Données spécifiques à Paris
  - 🗺️ Graphes de villes au format pickle: paris_graph.pkl, london_graph.pkl, rome_graph.pkl
- **[`src`](src)**: Implémentation principale
  - ⚙️ solver.py: Implémentation de la satisfaction de contraintes
  - 🏗️ city_generator.py: Générateur de données de ville
  - 🚕 distance_api.py: API pour calculer les temps de trajet
- [`plan_itinerary.py`](plan_itinerary.py): Script exécutable principal
- [`notebook.ipynb`](notebook.ipynb): Notebook Jupyter pour les démonstrations

---

## 📚 Références :
Vu et al., *Tourist Trip Design Problem with rich constraints (Branch-and-Check Approaches for the Tourist Trip Design Problem with Rich Constraints | Request PDF)】 – définit formellement le problème (objectifs et multiples contraintes pratiques : budget temps, horaires, catégories, ordre imposé) et propose une méthode exacte combinant sélection de lieux et vérification de chemin faisable. Souffriau et al. – travaux fondateurs sur le TTDP modélisé comme un Orienteering Problem avec contraintes, solution via une combinaison de CP et de VNS (Variable Neighborhood Search). Vansteenwegen, Orienteering Problem survey – discute des nombreuses variantes (Time Windows, Team orienteering, etc.) qui peuvent être toutes vues comme des ajouts de contraintes, ce qui milite pour une approche CSP modulable pour traiter un cas comme l’itinéraire touristique personnalisé.