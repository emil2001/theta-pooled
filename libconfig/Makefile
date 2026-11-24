SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o) libconfigcpp.o

all: libconfig++.a ../lib/libconfig.so

libconfig++.a: $(OBJECTS)
	@ar -cvq $@ $+

libconfig++.so: $(OBJECTS)
	@g++ -fPIC -shared $(OBJECTS) -o $@

../lib/libconfig.so: libconfig++.so
	@cp $< $@

libconfigcpp.o: libconfigcpp.c++
	@g++ -fPIC -c $< -o $@

%.o: %.c
	@gcc -std=gnu11 -fPIC -c $< -o $@

.PHONY: clean

clean:
	@rm -f *.o *.a *.so ../lib/libconfig.so
