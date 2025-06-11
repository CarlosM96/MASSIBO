-------------------------------------
-- Author: Ryan Rivera, FNAL			  
-- Created: Sep 11, 2015		
--  on Feb 3 -- BIG CHANGE, to not allow single byte address changing. Now a 32 bit reg in address space.
-- 	
-- This block takes decipher data on rising edge of capture_source_sig, if not 0 or 255	   
--
-- This allows the 1 byte packet ("CAPTAN Ping") to change the ip addrs of the GEI remotely
--
-- NOTE: reseting returns the GEI to the default address as specified by user logic.
-------------------------------------   


library IEEE;
use IEEE.std_logic_1164.all;		  		
use ieee.numeric_std.all;	   

	  		
use work.params_package.all;
																	
entity address_container is
	port (						 			 					
		clk : in std_logic;	 		   
		capture : in std_logic;										  
		data_in : in std_logic_vector(7 downto 0);			
											 
		set_ctrl_dest_strobe : out std_logic;	 
		set_data_dest_strobe : out std_logic;	
																			  		
		protocol_ping_strobe : out std_logic
		);
end;


architecture arch of address_container is	   

	signal capture_old : std_logic;		  							   

begin						  
		
	process(clk)
	begin
		
		if (rising_edge(clk)) then
			
			capture_old <= capture;			  
			protocol_ping_strobe <= '0';	  
			set_ctrl_dest_strobe <= '0';
			set_data_dest_strobe <= '0';
			
			if (capture_old = '0' and capture = '1') then   -- rising edge of source capture	
				if (data_in = x"00") then		-- 0s is used as "CAPTAN ping"	
					protocol_ping_strobe <= '1';									
				elsif (data_in = x"01") then		-- 1 is used to set ctrl dest with the current source
					set_ctrl_dest_strobe <= '1';									
				elsif (data_in = x"02") then		-- 2 is used to set data dest with the current source
					set_data_dest_strobe <= '1';									
				--else (data_in = x"FF")			-- 1s is used as "CAPTAN no-op"	
				end if;	   
			end if;	  
				
		end if;	
		
	end process;
	
end arch;