all:
	cd libhmm; make all
	cd pyhmm; make all

clean:
	cd libhmm; make clean
	cd pyhmm; make clean
