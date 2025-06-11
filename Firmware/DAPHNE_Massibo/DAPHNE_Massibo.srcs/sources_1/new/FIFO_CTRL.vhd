----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 04/08/2024 05:08:10 PM
-- Design Name: 
-- Module Name: FIFO_CTRL - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity FIFO_CTRL is
  Port ( 
    clka: in std_logic;
    reset: in std_logic;
    trig: in std_logic;
    wr_addr: in std_logic_vector(14 downto 0);
    fifo_full: in std_logic;
    fifo_empty: in std_logic;
    readable: out std_logic;
    we_ts: out std_logic;
    we: out std_logic;
    data_type:out std_logic_vector(1 downto 0)
  
  );
end FIFO_CTRL;

architecture Behavioral of FIFO_CTRL is

type state_type is (rst, wait4trig,readWaves, store, wait4done);
signal state: state_type;

type state_readable is (rst_read, fifo_readable,fifo_not_readable);
signal state_read: state_readable;

signal fifo_in: std_logic_vector(15 downto 0);
signal wave_count : std_logic_vector(9 downto 0);
signal waves_buffer : std_logic_vector(9 downto 0);
signal dt: std_logic_vector (1 downto 0);

signal we_sig: std_logic;
signal count : std_logic_vector(4 downto 0);


begin


    fifo_proc: process(clka)
    begin
        if rising_edge(clka) then
        
            if (reset='1') then
                wave_count <= "0000000000";-------
                we_sig   <= '0';
                state    <= rst;
                dt <= "00";
            else
                case state is
                    when rst =>
                        state <= wait4trig;
                        dt <= "00";
                    when wait4trig =>
                        wave_count <= "0000000000";
                        if (trig='1' and fifo_full='0' and we_sig ='0') then
                            state <= store;
                            we_sig <= '1';
                            dt <= "01";
                        else
                            state <= wait4trig;
                            we_sig <= '0';
                            wave_count <= "0000000000";
                            dt <= "00";
                        end if;
                        
                    when store =>
                  
                        if (fifo_full='1') then 
                            state <= wait4trig;--
                            --we <= '0';
                            --dt <= "00";
                            we_sig <= '1';
                            dt <= "11"; 
                        elsif (wave_count="0011111010") then -- 128 wave_count="1111111" 0011111010
                            state <= wait4trig;
                            we_sig <= '0';
                            dt <= "00";
                            
                        elsif (wave_count="0011111001") then-- wave_count="1111110" 0011111001
                            dt <= "11";      
                            state <= store;
                            wave_count <= std_logic_vector(unsigned(wave_count) + 1);
                            we_sig<= '1';              
                        else
                            state <= store;
                            wave_count <= std_logic_vector(unsigned(wave_count) + 1);
                            we_sig<= '1';
                            dt <= "10";     
                            

                            
                        end if;
                                      
                    when wait4done =>
                        if (trig='0') then
                            state <= wait4trig;
                        else
                            state <= wait4done;
                        end if;
                        dt <= "00";
                        
                    when others => 
                        state <= rst;    
    
                end case;
            end if;
        end if;
    end process fifo_proc;
   
    fifo_readble_proc: process(clka)
    begin
    
        if rising_edge(clka) then
        
                if (reset='1') then
                waves_buffer <="0000000000";
                readable   <= '0';
                state_read    <= rst_read;
                                

            else
            
                case state_read is
                    when rst_read =>
                        state_read <= fifo_not_readable;
                        readable   <= '0';
                        
                    
                    when fifo_not_readable =>
                    
                        readable   <= '0';    
                           
                        if (fifo_empty='1') then
                            state_read <= fifo_not_readable;
                            
                        
                        elsif (waves_buffer =  "0000000011" ) then  --7 waves buffer
                            state_read <= fifo_readable;                       
                            
                        elsif (wave_count = "0011111001"  ) then --1111111  0011111001
                            --state_read <= fifo_readable;
                            state_read <= fifo_not_readable;
                            waves_buffer <= std_logic_vector(unsigned(waves_buffer) + 1);                     
                        
                        end if;
                        
                    when fifo_readable =>
                        waves_buffer <="0000000000";
                    
                        if (fifo_empty='1') then
                            state_read <= fifo_not_readable;
                            readable <= '0';
                            
                        else
                            state_read <= fifo_readable;
                            readable <= '1';   
                            
                        end if;                 
                            
                    when others => 
                         state_read <= rst_read;                  
                end case;                  
            
            end if;
        end if;
    end process fifo_readble_proc;
    
    we_ts_proc: process(clka)
    begin
        if wave_count = "0011111001" then --wave_count="1111110"
            we_ts <= '1';
            
        else
            we_ts <= '0';
        end if;
    end process we_ts_proc;
    
    
    data_type <= dt;
    we <= we_sig;
    
    
    



end Behavioral;
