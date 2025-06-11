-- Author: Ryan Rivera, FNAL

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use IEEE.std_logic_unsigned.all;

entity or33 is
	port (					
		a1 : in std_logic;
		b1 : in std_logic;	 
		c1 : in std_logic;
		a2 : in std_logic; 			 
		b2 : in std_logic;			 
		c2 : in std_logic;
	
		
	
	
		ao		 		: out STD_LOGIC;
		bo		 		: out STD_LOGIC;
		co		 		: out STD_LOGIC
	) ;
end;


architecture or33_arch of or33 is
begin				  
	
	ao <= a1 or a2;
	bo <= b1 or b2;
	co <= c1 or c2;	
	
end or33_arch;
