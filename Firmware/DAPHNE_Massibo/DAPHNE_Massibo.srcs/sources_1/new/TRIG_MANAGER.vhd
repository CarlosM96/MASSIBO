----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 04/08/2024 02:38:28 PM
-- Design Name: 
-- Module Name: TRIG_MANAGER - Behavioral
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


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

library unisim;
use unisim.vcomponents.all;

library unimacro;
use unimacro.vcomponents.all;

use work.daphne_package.all;

entity TRIG_MANAGER is
  Port ( 
    top: in std_logic_vector(15 downto 0);
    mid: in std_logic_vector(15 downto 0);
    bot : in std_logic_vector(15 downto 0);
    mclk: in std_logic;
    oeiclk: in std_logic;
    trig_sync: in std_logic;
    reset: in std_logic;
    ext_trig: in std_logic;
    
    rx_addr: in std_logic_vector(31 downto 0);
    rx_wren: in std_logic;
    tx_rden: in std_logic;
    
    fifo_full: in std_logic_vector(2 downto 0);
    fifo_empty: in std_logic_vector(2 downto 0);
    
    threshold: in std_logic_vector(41 downto 0);
    
    top_wr_addr: in std_logic_vector (14 downto 0);
    mid_wr_addr: in std_logic_vector (14 downto 0);
    bot_wr_addr: in std_logic_vector (14 downto 0);
    
    top_out: out std_logic_vector(15 downto 0);
    mid_out: out std_logic_vector(15 downto 0);
    bot_out: out std_logic_vector(15 downto 0);
    
    
    

    readable: out std_logic_vector(2 downto 0);
    re: out std_logic_vector(2 downto 0);
    re_ts: out std_logic_vector(2 downto 0);
    we_ts: out std_logic_vector(2 downto 0);
    we: out std_logic_vector(2 downto 0)
  
  );
end TRIG_MANAGER;

architecture Behavioral of TRIG_MANAGER is

signal top0,top1,top2: std_logic_vector(13 downto 0);
signal mid0,mid1,mid2: std_logic_vector(13 downto 0);
signal bot0,bot1,bot2: std_logic_vector(13 downto 0);

signal counts_top,counts_mid,counts_bot : std_logic_vector(15 downto 0):=  (others=>'0');

signal self_trig, trig_sig, e_trig: std_logic_vector(2 downto 0);

signal trigger_we: std_logic;

--signal threshold_mode_reg : std_logic:='0';
signal threshold_mode_reg : std_logic_vector(1 downto 0) := (others=>'0');

signal threshold_array        : array_3x14_type; 

signal top_err_counter,mid_err_counter,bot_err_counter: std_logic_vector(31 downto 0) := (others=>'0');

signal dt_top,dt_mid,dt_bot: std_logic_vector(1 downto 0);

signal top_delay2, mid_delay2, bot_delay2: std_logic_vector(15 downto 0);


component FIFO_CTRL is
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
end component FIFO_CTRL;



begin
      gen_spy_signals: for b in 2 downto 0 generate
                threshold_array(b) <= threshold(((b)*14 + 13) downto ((b)*14));
      end generate gen_spy_signals;


    trig_pipeline_proc: process(mclk)
    begin
        if rising_edge(mclk) then
            top0 <= top(13 downto 0);   top1 <= top0;   top2 <= top1; 
            
            mid0 <= mid(13 downto 0);   mid1 <= mid0;   mid2 <= mid1; 
            
            bot0 <= bot(13 downto 0);   bot1 <= bot0;   bot2 <= bot1;
            
            
        end if;
    end process trig_pipeline_proc;

    self_trig(0) <= '1' when ( top2>threshold_array(0) and  top1<threshold_array(0)     and top0<threshold_array(0)     ) else '0';
    self_trig(1) <= '1' when ( mid2>threshold_array(1) and  mid1<threshold_array(1)     and mid0<threshold_array(1)     ) else '0';
    self_trig(2) <= '1' when ( bot2>threshold_array(2) and  bot1<threshold_array(2)     and bot0<threshold_array(2)     ) else '0';
    
    --Cruce positivo
--    self_trig(0) <= '1' when ( top2<threshold_array(0) and  top1>threshold_array(0)     and top0>threshold_array(0)     ) else '0';
--    self_trig(0) <= '1' when ( ext_trig = '1' ) else '0';
--    self_trig(1) <= '1' when ( mid2<threshold_array(1) and  mid1>threshold_array(1)     and mid0>threshold_array(1)     ) else '0';
--    self_trig(2) <= '1' when ( bot2<threshold_array(2) and  bot1>threshold_array(2)     and bot0>threshold_array(2)     ) else '0';
    
    
    
    counts_proc: process(mclk)
    
    begin
    
        if rising_edge(mclk) then
            if (self_trig(0) = '1') then
                counts_top <= std_logic_vector(unsigned(counts_top) + 1);
            end if;
            
            if (self_trig(1) = '1') then
                counts_mid <= std_logic_vector(unsigned(counts_mid) + 1);
            end if;

            if (self_trig(2) = '1') then
                counts_bot <= std_logic_vector(unsigned(counts_bot) + 1);
            end if;
            
        end if;
        
    end process counts_proc;
    


        FIFO_CTRL_TOP : FIFO_CTRL
        port map (
            clka => mclk,
            reset => reset,
            trig => trig_sig(0),
            fifo_full=>fifo_full(0),
            fifo_empty=>fifo_empty(0),
            readable=>readable(0),
            wr_addr => top_wr_addr,
            we_ts => we_ts(0),
            we => we(0),
            data_type=>top_out(15 downto 14)
        );
        FIFO_CTRL_MID : FIFO_CTRL
        port map (
            clka => mclk,
            reset => reset,
            trig => trig_sig(1),
            fifo_full=>fifo_full(1),
            fifo_empty=>fifo_empty(1),
            readable=>readable(1),
            wr_addr => mid_wr_addr,
            we_ts => we_ts(1),
            we => we(1),
            data_type=>mid_out(15 downto 14)
        );
        FIFO_CTRL_BOT : FIFO_CTRL
        port map (
            clka => mclk,
            reset => reset,
            trig => trig_sig(2),
            fifo_full=>fifo_full(2),
            fifo_empty=>fifo_empty(2),
            readable=>readable(2),
            wr_addr => bot_wr_addr,
            we_ts => we_ts(2),
            we => we(2),
            data_type=>bot_out(15 downto 14)
            
        );
        

    
    trigger_we <= '1' when ( (std_match(rx_addr,SELF_TRIGGER_MODE) OR std_match(rx_addr,SOFT_TRIGGER_MODE) OR std_match(rx_addr,EXT_TRIGGER_MODE))  and rx_wren='1'  ) else '0';

    trig_mode_proc: process(oeiclk)
    begin
        if rising_edge(oeiclk) then
            if (std_match(rx_addr,SOFT_TRIGGER_MODE)) then
                threshold_mode_reg <= "00";
            elsif ( std_match(rx_addr,SELF_TRIGGER_MODE) ) then
                threshold_mode_reg <= "01";
            elsif ( std_match(rx_addr,EXT_TRIGGER_MODE) ) then
                threshold_mode_reg <= "10";
            end if;
             
        end if;
        
    end process trig_mode_proc;
    
    
    e_trig_proc: process(mclk)
    begin
        if rising_edge(mclk) then
            e_trig(0) <= ext_trig;
            e_trig(1) <= ext_trig;
            e_trig(2) <= ext_trig;      
        end if;
        
    end process e_trig_proc;
    
    trig_out_proc: process(mclk)
    begin
        if rising_edge(mclk) then
            if (threshold_mode_reg = "00") then
                trig_sig <= (others=>trig_sync);
            elsif (threshold_mode_reg = "10") then
                trig_sig <= e_trig;
            else
                trig_sig <= self_trig;
            end if;
             
        end if;
        
    end process trig_out_proc;
    
    re(0) <= '1' when (std_match(rx_addr,FIFO_TOP_ADDR) and tx_rden='1') else '0'; 
    re(1) <= '1' when (std_match(rx_addr,FIFO_MID_ADDR) and tx_rden='1') else '0'; 
    re(2) <= '1' when (std_match(rx_addr,FIFO_BOT_ADDR) and tx_rden='1') else '0'; 
    
    re_ts(0) <= '1' when (std_match(rx_addr,FIFO_TOP_TS_ADDR) and tx_rden='1') else '0'; 
    re_ts(1) <= '1' when (std_match(rx_addr,FIFO_MID_TS_ADDR) and tx_rden='1') else '0'; 
    re_ts(2) <= '1' when (std_match(rx_addr,FIFO_BOT_TS_ADDR) and tx_rden='1') else '0'; 
    

    
    
    gendelay_TOP: for i in 13 downto 0 generate
        srlc32e_0_inst : srlc32e
        port map(
            clk => mclk,
            ce => '1',
            a => "11111",--"11111"
            d => top(i),
            q => top_delay2(i),
            q31 => open  
        );
        
    end generate gendelay_TOP;
        
   gendelay_TOP_2: for i in 13 downto 0 generate
        srlc32e_0_inst : srlc32e
        port map(
            clk => mclk,
            ce => '1',
            a => "11111",--"11111"
            d => top_delay2(i),
            q => top_out(i),
            q31 => open  
        );
    

    end generate gendelay_TOP_2;
    
    gendelay_MID: for i in 13 downto 0 generate
        srlc32e_0_inst : srlc32e
        port map(
            clk => mclk,
            ce => '1',
            a => "11111",
            d => mid(i),
            q => mid_delay2(i),
            q31 => open  
        );
    

    end generate gendelay_MID;
    
     gendelay_MID_2: for i in 13 downto 0 generate
        srlc32e_0_inst : srlc32e
        port map(
            clk => mclk,
            ce => '1',
            a => "11111",
            d => mid_delay2(i),
            q => mid_out(i),
            q31 => open  
        );
    

    end generate gendelay_MID_2;
    
    gendelay_BOT: for i in 13 downto 0 generate
        srlc32e_0_inst : srlc32e
        port map(
            clk => mclk,
            ce => '1',
            a => "11111",
            d => bot(i),
            q => bot_delay2(i),
            q31 => open  
        );
    

    end generate gendelay_BOT;
    
    gendelay_BOT_2: for i in 13 downto 0 generate
        srlc32e_0_inst : srlc32e
        port map(
            clk => mclk,
            ce => '1',
            a => "11111",
            d => bot_delay2(i),
            q => bot_out(i),
            q31 => open  
        );
    

    end generate gendelay_BOT_2;

    

end Behavioral;
