# Objekterkennung
Bei der Suche nach einem Objekterkennungsmodell habe ich mich nach nur kurzer Recherche für eines der Yolo Modelle von ultralytics entschieden, da diese in den letzten Jahren für sehr viel Aufsehen in der Szene gesorgt haben. Diese Modelle sind für die private oder akademische Nutzung kostenlos.
Ich habe mich für die neuste Version deren Modell entschieden, was zu Zeitpunkt der Entwicklung die V11 war.

## Training
Bevor ich trainieren konnte, musste ich Bilder Labeln. Als Datensatz habe ich die Bilder verwendet, welche mit einer Smartphonekamera aufgenommen wurden, wobei versucht wurde, möglichst viele Betrachtungswinkel und Entfernungen vom Objekt abzudecken.
Ich habe mich für die Labelling Software "label-studio" entschieden. Dabei handelt es sich über eine lokal installierbare Software mit einer Weboberfläche basierend auf python. Diese Software ermöglicht es, viele Bilder exakt zu labeln und verschiedenen Objektklassen zuzuordnen. Außerdem unterstützt diese Software den Export der Label dateien im "Yolo Format". Dies speichert alle Labels in passenden textdateien und die dazugehörigen Bilder in korrekt benannten Ordnern ab, was eine große Erleichterung war. Ich habe 113 Bilder gelabelt, was sich vor allem für die Nano Modellgröße als ausreichend herausgestellt hat.
Nachdem die labels fertiggestellt waren, habe ich mich damit beschäftigt, den Trainingsprozess möglichst effizient zu gestalten. Da ich einen Rechner mit dedizierter Grafikkarte von NVIDIA besitze, konnte ich die Cuda Beschleunigung nutzen. Damit dies funktioniert hat, musste ich ein Cuda Paket installieren, und auch eine Torch version herunterladen, welche passend zu meinem Grafikkartentreiber war. Dies hat einige Zeit in Anspruch genommen, hat mir dann aber das deutlich schnellere Trainieren ermöglicht.

## Objekterkennung
https://www.ultralytics.com/blog/the-evolution-of-object-detection-and-ultralytics-yolo-models
### Grundlagen der Objekterkennung
Die Objekterkennung kombiniert zwei Aufgaben: die Lokalisierung von Objekten in einem Bild und deren Zuweisung zu einer Objektklasse. Klassische Ansätze basierten auf Feature-Engineering, wie HOG (Histogram of Oriented Gradients) oder SIFT, kombiniert mit SVMs oder Entscheidungsbäumen. Diese Methoden waren jedoch langsam und wenig robust.

Mit der Einführung von tiefen neuronalen Netzen in und insbesondere Convolutional Neural Networks (CNNs) in den 2010er Jahren wurde die Objekterkennung revolutioniert. Moderne Modelle wie YOLO, SSD (Single Shot Detector) oder Faster R-CNN erledigen beide Aufgaben – Erkennung und Lokalisierung – in einem einzigen, trainierbaren Netzwerk.

### YOLO: You Only Look Once
YOLO wurde ursprünglich entwickelt, um die Objekterkennung extrem schnell und ganzheitlich zu machen. Der Name „You Only Look Once“ verweist darauf, dass das gesamte Bild in einem einzigen Durchlauf verarbeitet wird. Im Gegensatz zu früheren Architekturen, die das Bild mehrfach analysierten oder vorschlagsbasiert vorgingen, betrachtet YOLO das Bild als Ganzes und gibt direkt pro Bildregion Vorhersagen aus.

Das Eingabebild wird in ein Raster (Grid) unterteilt. Jeder Zellenbereich dieses Rasters ist dafür verantwortlich, Objekte zu erkennen, deren Mittelpunkt in diesem Bereich liegt. Pro Zelle sagt das Modell eine oder mehrere Bounding Boxes mit zugehörigen Klassenzugehörigkeiten und Konfidenzwerten vorher.

![Yolo funktionsweise](BilderBericht\YoloOverview.png)

Dadurch dass die YOLO Modellreihe als open source Projekt einsehbar ist, wurde das Modell ständig weiterentwickelt und es konnten Ansätze von Entwicklern aus der ganzen Welt eingebracht werden.

# Auslesen der Brille (Beispielüberschrift)

# Übertragung (Beispielüberschrift)

# Webseite (Beispielüberschrift)

# get Started (Beispielüberschrift)