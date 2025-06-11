-- Author: Ryan Rivera, FNAL

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use IEEE.std_logic_unsigned.all;

entity dataout_mux is
	port (					
		sel_udp : in std_logic;	 		  
		
		arp_tx_en 			: in STD_LOGIC;
		arp_tx_er 			: in STD_LOGIC;
		arp_data_out 		: in STD_LOGIC_VECTOR(7 downto 0); 
		
		udp_tx_en 			: in STD_LOGIC;
		udp_tx_er 			: in STD_LOGIC;
		udp_data_out 		: in STD_LOGIC_VECTOR(7 downto 0);
								
		tx_en		 		: out STD_LOGIC;
		tx_er		 		: out STD_LOGIC;  		
		txd 				: out STD_LOGIC_VECTOR(7 downto 0)
	) ;
end;


architecture dataout_mux_arch of dataout_mux is		    		  
begin				  
	
	tx_en	  <= udp_tx_en		when sel_udp = '1' else	arp_tx_en; 
	tx_er	  <= udp_tx_er 		when sel_udp = '1' else	arp_tx_er; 
	txd 	  <= udp_data_out 	when sel_udp = '1' else arp_data_out; 	
	
	
end dataout_mux_arch;
