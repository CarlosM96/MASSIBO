-- Author: Ryan Rivera, FNAL

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use IEEE.std_logic_unsigned.all;

entity user_addrs_mux is
	port (											  	  
		user_length : in std_logic_vector(10 downto 0);	 	
		user_dest_addr : in std_logic_vector(31 downto 0);	 
		ping_mode : in std_logic;					    
		
		icmp_mode : in std_logic;		
		icmp_length : in std_logic_vector(10 downto 0);		 
		icmp_dest_addr : in std_logic_vector(31 downto 0);
															
		ip_dest_addr : out std_logic_vector(31 downto 0);
		ip_tx_length : out std_logic_vector(10 downto 0)
	) ;
end;


architecture user_addrs_mux_arch of user_addrs_mux is
begin				  						
															 		  
	ip_tx_length <=  						 --"000" & x"05"; -- "101" & x"C0";	 -- range: 0 to x5C0 (1472) 
		"000" & x"02" when ping_mode = '1' else	   	
		icmp_length when icmp_mode = '1' else	 
		user_length;   									 		  
	ip_dest_addr <=  						 	   	
		icmp_dest_addr when icmp_mode = '1' else	 
		user_dest_addr;
	
end user_addrs_mux_arch;				