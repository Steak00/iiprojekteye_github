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

# Auslesen und Verarbeitung der Brillendaten
## Funktionsweise Pupil Capture
Pupil Capture ist eine Softwareplattform von Pupil Labs, die speziell für Eye-Tracking-Anwendungen entwickelt wurde. Sie ermöglicht die Erfassung, Verarbeitung und Visualisierung von Blickdaten, indem sie Daten von speziellen Eye-Tracking-Brillen oder Kameras aufzeichnet. Die Software erkennt die Augenbewegungen des Nutzers und berechnet daraus präzise den sogenannten Gaze-Punkt, also den Punkt, auf den der Nutzer gerade schaut.  
Die Funktionsweise von Pupil Capture basiert auf mehreren Komponenten: Zunächst werden über Kameras die Augen und das Sichtfeld des Nutzers aufgenommen. Anschließend verarbeitet die Software die Bilddaten in Echtzeit, erkennt die Pupillen und bestimmt deren Position. Durch komplexe Algorithmen wird daraus der Blickvektor berechnet, der auf ein Bild oder eine Szene projiziert wird, um den exakten Blickpunkt zu ermitteln.  
Pupil Capture bietet eine modulare Architektur mit Plugins, die unterschiedliche Funktionen ermöglichen, wie etwa die Kalibrierung, Aufzeichnung oder das Streaming von Daten. Besonders wichtig ist das Plugin „Network API“, welches den Zugriff auf die Rohdaten über ein Netzwerkprotokoll erlaubt. So können externe Anwendungen per TCP-Verbindung live auf die Gaze-Daten und Kamerabilder zugreifen.  
Diese Echtzeit-Datenübertragung wird häufig genutzt, um Blickbewegungen in interaktiven Anwendungen auszuwerten, Forschungsdaten zu sammeln oder Assistenzsysteme zu entwickeln. Die Kombination aus präziser Erfassung und flexibler Schnittstelle macht Pupil Capture zu einem mächtigen Werkzeug für Eye-Tracking-Projekte.  

Quelle: https://docs.pupil-labs.com/core/software/pupil-capture/

## Funktionsweise Python-Skript zur Verarbeitung der Brillen-Daten
[Zur Datei](skript_brille/skript_brille.py)  
### Vorbereitung
Zu Beginn habe ich kleinere Experimente mit der Brille durchgeführt, um die Funktionsweise zu testen. Nachdem das erfolgt ist, habe ich nach Methoden gesucht wie man die ausgelesenen Daten, vor allem den Gaze Punkt, von der Brille in das Python Skript übertragen kann. 
Der Gaze Punkt ist der Blickpunkt der Person, das heißt dadurch kann man darauf schließen wohin der Nutzer gerade schaut und welches Objekt betrachtet wird.  

### Datenübertragung
Die Daten der Brille wurden über die integrierte Network API der Software Pupil Capture exportiert. Man aktiviert dafür in der Software im Plugin Manager den Plugin „Network API“, dieser stellt dann die benötigten Daten über einen TCP-Stream bereit, welcher später in unserer Anwendung eingelesen werden kann.  

Zur Kommunikation zwischen Pupil Capture und unserer Anwendung wurde das ZeroMQ-Messaging-Framework genutzt, dieses ist für hochperformante, asynchrone Datenübertragung zwischen Prozessen ausgelegt, was in unserem Fall gebraucht wurde. Mithilfe von diesem Framework wurde eine Echtzeit-Kommunikation zwischen unserer Anwendung und der Eye-Tracking-Software Pupil Capture aufgebaut.  
Die Verbindung zur Pupil-Capture-Instanz wird über das TCP-Protokoll realisiert, das von ZeroMQ (zmq) intern verwendet wird. Damit konnten wir über unseren Python-Client gezielt Steuerbefehle senden und verschiedene Datenströme, wie z.B. die Gaze-Daten und die Videoframes der Weltkamera empfangen.
Mithilfe von zmq konnten wir verschiedene Sockettypen erstellen, die sich gezielt mit dem Senden oder dem Empfang von Daten beschäftigen.
Anfangs haben wir einen zmq.Context erstellt, dieser fungiert als Container und Lebenszyklus-Manager für alle im weiteren Verlauf erzeugten Sockets. 
Anschließend wird eine initiale Verbindung zu localhost auf Port 50020 hergestellt. Dabei handelt es sich um die standardmäßige TCP-Adresse, unter der die Pupil-Capture-Software die Anfragen entgegennimmt.  

Pupil Capture verwendet ein Publish/Subscribe-Modell, um die Sensordaten wie die Gazedaten oder Kameraframes über einen separaten Port zu streamen. Da dieser Port dynamisch zugewiesen wird, muss der Client ihn zunächst über einen REQ-Socket erfragen.  
Hierfür wurde ein REQ-Socket erzeugt, der sich mit dem Steuerport verbindet. Mit einem speziellen Befehl wird dann vom Client der aktuelle Datenstream-Port abgefragt. Die Antwort von Pupil Capture enthält dann den tatsächlichen Port über welchen die Daten veröffentlicht werden.
Um die gewünschten Daten empfangen zu können, müssen bestimmte Plugins in Pupil Capture aktiv gestartet werden, dies geschieht durch das Senden der entsprechenden Befehle (‚start_plugin gaze‘, ‚ start_plugin gaze_streaming‘). Ich habe die Befehle zur Verarbeitung und Veröffentlichung der Gaze Daten an die Software geschickt.  
Außerdem wurden zwei unabhängige Subscriber-Sockets eingerichtet um zum einen die Gaze-Daten und zum anderen die Kamerabilder der Weltkamera separat empfangen zu können.  
Beide Sockets waren dann bereit die Daten zu empfangen, sobald diese über das Netzwerk publiziert werden.  

### Empfang der Daten
Zum Händeln des Empfangs der Gaze Daten wurde ein Listener implementiert. Dieser kümmert sich ausschließlich um die Verarbeitung der empfangenen Nachrichten zu den Gaze-Daten. Die Nachrichten werden eingelesen und die benötigten konkreten Daten für die Weiterverarbeitung extrahiert, außerdem erfolgt eine Fehlerbehandlung. Diese Funktion hält die aktuellen Daten zur Blickposition jederzeit aktuell und macht sie zugreifbar für andere Programmteile.  

Auch für den Empfang der Kamerabilder der Weltkamera wurde ein separater Listener implementiert, dieser validiert, liest und decodiert die eingehenden Nachrichten ähnlich wie der Gaze-Listener und stellt das aktuelle Kamerabild jederzeit zur Weiterverarbeitung bereit.  

### Objekterkennung
Zur Objekterkennung wurde eine Funktion implementiert. Diese führt die Objekterkennung kontinuierlich durch. Dafür nutzt sie unser eigens trainiertes YOLO-Modell und wendet das auf den empfangenen Kamerabilder an um die erkannten Objekte zu identifizieren und deren Positionen zu extrahieren.  
Die erkannten Objekte werden dann in einer gemeinsamen Variable gespeichert, sodass andere Programmteile darauf zugreifen können. Diese Funktion läuft in einem separaten Hintergrundthread und arbeitet parallel zu den beiden Listenern für die Kamera- und die Gaze-Daten.

### Gesammelte Verarbeitung und Versenden über MQTT
Eine while-Schleife bildet den zentralen Verarbeitungsschritt. Dort werden immer die aktuellen Gaze-Daten, die Kamera-Daten und die erkannten Objekte eingelesen. Danach wird geprüft, ob die Daten gültig sind und die empfangenen normierten Daten werden auf reale Pixelwerte umgerechnet. Daraufhin wird geprüft, ob der Gazepunkt innerhalb einer erkannten Bounding Box liegt, falls ja wird das Objekt als „angesehen“ markiert.   

Als Zwischenschritt und zur Überprüfung wurde ein Codeteil eingebaut mit dem man sich den empfangenen Stream und die gelabelten Objekte mit Beschreibung live ansehen konnte, dieser wurde in der finalen Version entfernt, um die Verarbeitungszeit zu minimieren.  

Anschließend wird überprüft, ob sich das aktuelle Objekt vom vorherigen Unterscheidet. Bei einer Unterscheidung wird das neue Objekt über MQTT publiziert. Diese Daten können dann wiederum von der Webseite eingelesen werden. Für die MQTT Verbindung wurde zu Beginn ein MQTT-Client definiert, der sich mit dem entsprechenden Broker verbindet und die Nachrichten auf ein fest definiertes Topic published. Zwischen den einzelnen Schleifenwiederholungen ist eine Pause von 10ms eingebaut und es ist eine Fehlerbehandlung integriert.

### Separate Effizientere Version
[Zur Datei](skript_brille/skript_brille_optimized.py)  

Die hier beschriebene Version, welche zur Bilderkennung immer den ganzen übermittelten Frame benutzt stellte sich als sehr rechenintensiv heraus, deswegen wurde ein optimiertes Skript für die Brille implementiert, dieses verarbeitet ausschließlich einen Ausschnitt um den Gaze-Punkt und nutzt diesen zur Objekterkennung. Die Implementierung erfolgte nach Rückgabe der Brille, deswegen konnten wir ihn leider nicht mehr testen.



# Übertragung zwischen Brille und Webseite

Zur Übertragung wird das MQTT Protokoll genutzt. Dafür wird in dem Startskript ein Eclipse MQTT Broker erstellt, der die Nachrichten zwischen dem Skript für die Brille und der Webseite übermittelt. Außerdem wurde hierfür noch ein neues Topic angelegt: "eye_tracking/detected_object".  

MQTT (Message Queuing Telemetry Transport) wird in diesem Projekt verwendet, um eine leichte, effiziente und echtzeitfähige Kommunikation zwischen dem Eye-Tracking-System und der Webanwendung zu gewährleisten. Das Eye-Tracking-Skript erkennt und klassifiziert die Objekte, auf die der Nutzer blickt, und sendet diese Information über einen MQTT-Broker an die Webseite. MQTT eignet sich besonders gut, weil es ein leichtgewichtiges Protokoll ist und daher auch bei geringer Bandbreite zuverlässig funktioniert. Außerdem wird dadurch eine asynchrone Nachrichtenübertragung ermöglicht mittels Publish7Subscribe und es lässt sich sehr einfach in viele Systeme integrieren. Dadurch kann z. B. die Webseite immer aktuell anzeigen, welches Objekt gerade angesehen wird, ohne dass eine direkte Verbindung oder permanente Abfrage notwendig ist.  

# Webseite (Beispielüberschrift)

# get Started (Beispielüberschrift)
