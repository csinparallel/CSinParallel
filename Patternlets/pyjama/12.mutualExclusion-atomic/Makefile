PROG  = Atomic
CLASS = $(PROG).class
SRC   = $(PROG).java
PJ    = ../lib/Pyjama.jar
CC    = java -jar $(PJ)
VM	  = java -cp $(PJ):. 

run: $(CLASS)
	$(VM) $(PROG) $(ARGS)

$(CLASS): $(SRC)
	$(CC) $(SRC)

clean:
	rm -f $(PROG)*class *~ *#
