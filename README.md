# Skrypt do monitorowania smogu w okolicach miasta Rybnik (Python)

Airly [1] to firma, budująca sieć sensorów powietrza dla miast i gmin. Umożliwia ona monitorowanie stanu powietrza w czasie rzeczywistym za pomocą mapy online oraz udostępnia api, które pozwala na ściągnięcie dokładnych danych dotyczących interesujących nas regionów.
Rybnik to jedno z najbardziej zanieczyszczonych miast w Polsce, jesienią zeszłego roku, w każdej dzielnicy zostały zainstalowane sensory powietrza, razem jest ich 27.
	
	Celem skyptu jest:
	 1) pobieranie codzienne danych dotyczących jakości powietrze w każdej z dzielnic 
	 2) opracowanie wyników w formie przystępnych tabel i mapy
	 3) wrzucenie wyników na stworzone wcześniej strony w portalach społecznościowych, np. Facebook
	
	
1. Dostęp do danych możliwy jest przy wykorzystaniu cURL. Pythonowa biblioteka Requests [2] pozwala na wygodne wykorzystanie wspomnianego powyżej api

2. Następne kroki to processing ściągniętych danych do wygodnej i przejrzystej formy. Jego wynikiem będą tabele z danymi, wykresy i ostatecznie mapa z zanieczyczeniem w każdej dzielnicy.
Wykorzystane zostaną przede wszystkim biblioteki Pandas, Matplotlib, PySAL (lub inna biblioteka wspierająca tworzenie map)

3. Następnie dane zostaną wrzucone na stworzone wcześniej strony w portalach społecznościowych, (np. Facebook, Tweeter) przy wykorzystaniu odpowiednich api [3] https://www.facebook.com/powietrzewrybniku/


[1] https://airly.eu/en/

[2] http://docs.python-requests.org/en/master/

[3] http://facebook-sdk.readthedocs.io/en/latest/
