-- Author: Ryan Rivera, FNAL

library IEEE;
use IEEE.std_logic_1164.all;	

entity icmp_ping_shift_reg is
	port (					
		clk : in std_logic;	 	   	 
		us_clken : in std_logic;	 
		ds_clken : in std_logic;	 
											
		din 				: in STD_LOGIC_VECTOR(7 downto 0);		
		dout 				: out STD_LOGIC_VECTOR(7 downto 0)
	) ;
end;


architecture arch of icmp_ping_shift_reg is   
		 
	constant SHR_DEPTH : natural := 44; -- was 33, but was not considering dest addr or type (so extraneous pings replies were occuring)											 	  								   
	type shReg_t is array(natural range <>) of std_logic_vector(7 downto 0);  
	signal shReg 						: shReg_t(SHR_DEPTH-1 downto 0);		
	signal	din_latch 				: STD_LOGIC_VECTOR(7 downto 0);
	
begin				  
	
	dout <= shReg(SHR_DEPTH-1);
	process(clk)
	begin
		if rising_edge(clk) then
			if us_clken = '1' then 
				din_latch <= din;
			end if;
			
			if ds_clken = '1' then
				shReg <= shReg(SHR_DEPTH-2 downto 0) & din_latch;	
			end if;		
		end if;
	end process;
	
end arch;
