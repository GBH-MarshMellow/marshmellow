package com.gbh.gbh_mm.finance.bank.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import java.util.Map;

public interface BankFinService {

//    String test();
    Map<String, Object> findBankList() throws JsonProcessingException;
}
