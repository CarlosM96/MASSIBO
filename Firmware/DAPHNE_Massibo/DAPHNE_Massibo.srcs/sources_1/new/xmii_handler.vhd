-------------------------------------
-- Author: Ryan Rivera, FNAL			  
-- Created: Sep 4, 2015 																							   
-- 
-- handles RGMII packets NOT at all. Expects GMII, so just passthrough.			  										 
--																						  
-------------------------------------  
							  
----------------------------- 		                                                                                          
--                                                                                                                              
--  Comments by rrivera at fnal dot gov : Sep 4, 2015                                                                          
--                                                                                                                              
--  IMPORTANT!! IMPORTANT!! 				                                                                                      
--  It's very important to note!!!   	                                                                                          
--  										                                                                                      
--  The script that moves these files into a Firmware project will 		                                                      
--   substitue this rgmii_handler in if parameter 2 is MII_100_1000		
			 				   

library IEEE;
use IEEE.std_logic_1164.all;		  		
use ieee.numeric_std.all;		  
								
																	
entity MII_100_1000_handler is
	port (						 			 					
		clk : in std_logic;	 	  
		reset : in std_logic;	  
		
		-- rx
		rx_data : in std_logic_vector(7 downto 0);	
		rx_dv : in std_logic;	  
		rx_er : in std_logic;	     	   		 
		
		rx_data_handled : out std_logic_vector(7 downto 0);	   
		rx_dv_handled : out std_logic;				  
		rx_er_handled : out std_logic;	
		
		-- tx
		tx_data : in std_logic_vector(7 downto 0);	
		tx_dv : in std_logic;	  
		tx_er : in std_logic;	 						   
		
		tx_data_handled : out std_logic_vector(7 downto 0);				  
		tx_dv_handled : out std_logic;				  
		tx_er_handled : out std_logic
															 
	) ;
end;


architecture arch of MII_100_1000_handler is	   
begin				  	  
									   
	rx_data_handled <=	 	rx_data;
	rx_dv_handled 	<=		rx_dv;										  
	rx_er_handled 	<= 	 	rx_er;	  
	
								   	   
	
	tx_data_handled	<=	 	tx_data;
	tx_dv_handled 	<=		tx_dv;	
	tx_er_handled 	<= 	 	tx_er;		  
	   
end arch;					 
                --
--
--entity RX_IN_LATCH is
--    Port ( clk : in  STD_LOGIC;
--           dv : in  STD_LOGIC;
--           er : in  STD_LOGIC;
--           d : in  STD_LOGIC_VECTOR (7 downto 0);
--           dvo : out  STD_LOGIC;
--           ero : out  STD_LOGIC;
--           do : out  STD_LOGIC_VECTOR (7 downto 0));
--end RX_IN_LATCH;
--
--architecture Behavioral of RX_IN_LATCH is	   
--       signal    dvs :  STD_LOGIC;
--       signal    ers :  STD_LOGIC;
--       signal    ds :   STD_LOGIC_VECTOR (7 downto 0);
--
--begin
--
--	process(clk)
--	begin
--		if falling_edge(clk) then
--			dvs <= dv;
--			ers <= er;
--			ds <= d;
--		end if;
--	end process; 
--	
--	process(clk)
--	begin
--		if rising_edge(clk) then
--			dvo <= dvs;
--			ero <= ers;
--			do <= ds;
--		end if;
--	end process;
--
--end Behavioral;


















