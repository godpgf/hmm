LIB_DIR := ../lib/

OBJECTS := libhmm_api.o

all: libhmm

libhmm: $(OBJECTS)
	mkdir -p $(LIB_DIR)
	ar crv $(LIB_DIR)libhmm_api.a $(OBJECTS)
	g++ -fPIC -std=c++11 -shared -o $(LIB_DIR)libhmm_api.so $(OBJECTS)

%.o: %.cpp
    #g++ -fPIC -std=c++11 -O3 -Wall -c -g -rdynamic $< -o $@
	g++ -fPIC -std=c++11 -O3 -Wall -c -g -rdynamic $< -o $@

clean:	clean_lib
	mkdir -p $(LIB_DIR)
	rm -rf $(LIB_DIR)

clean_lib:
	rm -f $(OBJECTS)



