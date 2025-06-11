----------------------------------------------------------------------------------
-- Company:  FNAL
-- Engineer:  Ryan Rivera
-- 
-- Create Date:    16:52:08 12/04/2007 	 
-- Dependencies: 
--
-- Revision: 
-- Revision 0.01 - File Created
-- Additional Comments: 
--
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

entity crc_splice is
    Port ( 
			data 				: in  STD_LOGIC_VECTOR (7 downto 0);
			crc 				: in  STD_LOGIC_VECTOR (7 downto 0);
			rd 					: in  STD_LOGIC;			  
		   									 	  	  
			clk 				: in STD_LOGIC;
			tx_en_in 			: in STD_LOGIC;
			tx_er_in 			: in STD_LOGIC;
			four_bit_mode		: in STD_LOGIC;
											  
			crc_mask			: out STD_LOGIC;
			tx_en		 		: out STD_LOGIC;
			tx_er		 		: out STD_LOGIC;  		
			txd 				: out STD_LOGIC_VECTOR(7 downto 0)
		);
end entity;

architecture arch of crc_splice is
						
     signal  dataout : STD_LOGIC_VECTOR (7 downto 0);  	
     signal  nibble_reg : STD_LOGIC_VECTOR (3 downto 0);  --crc doesn't hold output, so need to latch
	 signal	flip_flop 		 		: STD_LOGIC;  
begin
	
	crc_mask		 	<= not flip_flop;
							
	dataout	  			<= crc when rd = '1' else data; 	 
		
	process(clk)
	begin
		
		if rising_edge(clk) then	    
			tx_en  <= tx_en_in;
			tx_er  <= tx_er_in;
			
			if four_bit_mode = '0' then	 
				txd   <= dataout; 		
				flip_flop <= '0';	--disable mask	
			else  		  -- in four bit mode alternate nibbles during tx_en = 1
				if tx_en_in = '0' then	 
					flip_flop <= '0';
				else
					flip_flop <= not flip_flop;
				end if;	 
				
				if flip_flop = '0' then	  
					txd(3 downto 0) <= dataout(3 downto 0);	  
					nibble_reg <= dataout(7 downto 4); 	--forced by crc gen block
				else			
					txd(3 downto 0) <= 	nibble_reg;			 
				end if;	 
				
			end if;
		end if;
	end process;

end arch;

