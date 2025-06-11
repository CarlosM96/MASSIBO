-- Author: Ryan Rivera, FNAL

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use IEEE.std_logic_unsigned.all;

entity udp_data_splicer is
	port (	
		clk : in std_logic;
		user_data : in std_logic_vector(7 downto 0);
		gen_data : in std_logic_vector(7 downto 0);
		
		sel_user : in std_logic;   
		
		udp_data_out : out std_logic_vector(7 downto 0)
	) ;
end;


architecture udp_data_splicer_arch of udp_data_splicer is				   

	signal delay_sel_user : std_logic;		
	signal latched_user_data, weird_sim_fix : std_logic_vector(7 downto 0);		 
	
begin				  

	udp_data_out <= gen_data when delay_sel_user = '0' else latched_user_data;
	
	weird_sim_fix <= user_data after 1ns; -- simulation clock is confused,.. so force clock latch to work.
	
	process(clk)
	begin
		if rising_edge(clk) then	  
			latched_user_data <= weird_sim_fix;
			delay_sel_user <= sel_user;
		end if;				   
	end process;	
	
end udp_data_splicer_arch;				