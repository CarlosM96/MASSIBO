-------------------------------------
-- Author: Ryan Rivera, FNAL			  
-- Created: Jan 29, 2016																							  
-- 	
-- xFAA6 is the one's complement checksum (non-inverted) without the dest and src

-- x85F9 is the checksum (non-inverted) without
-- the data length and lower address word with. Include extra 0x1C for header length
--increase above input length
--
--ID := x3579,
--vers/header := x4500,
--ToS := 0,
--Flags/Frag := 0,
--ttl/protocol := x8011,
--dest := xC0 A8 85 __
--src := xC0 A8 85 __		   

--if (icmp_mode = '0') then -- normal udp mode
--	cs_sig <= '0' & x"FAA6"; -- was 85F9 -- ttl/protocol := x8011
--else					  -- icmp ping mode
--	cs_sig <= '0' & x"FA96"; -- was 95E9 -- ttl/protocol := x8001
--end if;							

-- correct result is 79FA for source 2 dest 0 on 192.168.133.X
-------------------------------------   


library IEEE;
use IEEE.std_logic_1164.all;		  		
use ieee.numeric_std.all;	   

	  		
use work.params_package.all;
																	
entity ip_checksum_calc is
	port (						 			 					
		clk : in std_logic;	 	  
		reset : in std_logic;		   
		trigger : in std_logic;		
		
		icmp_mode : in std_logic;	
		
		src_in : in std_logic_vector(31 downto 0);				  
		dest_in : in std_logic_vector(31 downto 0);				  
		length_in : in std_logic_vector(10 downto 0);		
		
		cs : out std_logic_vector(15 downto 0)
		);
end;


architecture arch of ip_checksum_calc is	   

	signal trigger_old : std_logic;
	signal cs_sig : unsigned(16 downto 0) := (others => '0');	  
	signal add_sig : unsigned(16 downto 0) := (others => '0');	 
	signal state : unsigned(3 downto 0) := (others => '0');	 

begin			 			   
	cs <= std_logic_vector(not cs_sig(15 downto 0));
		
	process(clk)
	begin
		
		if (rising_edge(clk)) then
			
			trigger_old <= trigger;	  				
			
			-- always be adding one's complement over 2 state counts
			if(state(0) = '1') then
				cs_sig <= cs_sig + add_sig;
			elsif(cs_sig(16) = '1') then
				cs_sig(15 downto 0) <= cs_sig(15 downto 0) + 1;
  				cs_sig(16) <= '0';
			end if;				
			
			
			-- control state counter
			if (reset = '1' or (trigger_old = '0' and trigger = '1')) then	-- reset state
				state <= (others => '0');	
				add_sig <= (others => '0');	
				
				if (icmp_mode = '0') then -- normal udp mode
					cs_sig <= '0' & x"FAA6"; -- was 85F9 -- ttl/protocol := x8011
				else					  -- icmp ping mode
					cs_sig <= '0' & x"FA96"; -- was 95E9 -- ttl/protocol := x8001
				end if;		
			elsif(state < 9) then
				state <= state + 1;						
			end if;
				  
			
			-- states
			if(	state = 0 ) then
				add_sig <= '0' & unsigned(src_in(15 downto 0));	 
			elsif(	state = 2 ) then
				add_sig <= '0' & unsigned(src_in(31 downto 16));	 
			elsif(	state = 4 ) then
				add_sig <= '0' & unsigned(dest_in(15 downto 0));
			elsif(	state = 6 ) then
				add_sig <= '0' & unsigned(dest_in(31 downto 16)); 
			elsif(	state = 8 ) then
				add_sig <= '0' & '0' & x"0" & unsigned(length_in(10 downto 0));  			
			elsif(	state > 8 ) then
				add_sig <= (others => '0'); -- add nothing
			end if;
				
		end if;	
		
	end process;
	
end arch;