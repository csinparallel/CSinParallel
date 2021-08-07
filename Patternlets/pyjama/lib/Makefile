URL  = https://www.csc.tntech.edu/pdcincs/resources/modules/tools/updated/Pyjama.zip
PJ    = Pyjama
JAR   = $(PJ).jar

$(PJ):
	wget $(URL)
	unzip -o $(PJ).zip
	cp $(PJ)/*.jar $(JAR)
	rm -r -f Pyjama.zip Pyjama

clean:
	rm $(JAR)
	

