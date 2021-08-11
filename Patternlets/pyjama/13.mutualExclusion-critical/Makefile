PROG  = Critical
CLASS = $(PROG).class
SRC   = $(PROG).java
PJ    = ../lib/Pyjama.jar
CC    = java -jar $(PJ)
VM	  = java -cp $(PJ):. 
GEN   = gen

run: $(CLASS)
	$(VM) $(PROG) $(ARGS)

$(CLASS): $(SRC)
	$(CC) $(SRC)

clean: 
	rm -f $(PROG)*class *~ *#
	rm -r -f $(GEN)

gen:
	mkdir gen
	
j2j: gen
	$(CC) -j2j -d $(GEN) $(SRC)

