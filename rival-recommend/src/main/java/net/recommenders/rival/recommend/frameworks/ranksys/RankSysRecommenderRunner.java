/*
 * Copyright 2015 recommenders.net.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.recommenders.rival.recommend.frameworks.ranksys;

import es.uam.eps.ir.ranksys.core.Recommendation;
import static es.uam.eps.ir.ranksys.core.util.parsing.DoubleParser.ddp;
import static es.uam.eps.ir.ranksys.core.util.parsing.Parsers.lp;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import es.uam.eps.ir.ranksys.mf.Factorization;
import es.uam.eps.ir.ranksys.mf.Factorizer;
import es.uam.eps.ir.ranksys.mf.als.HKVFactorizer;
import es.uam.eps.ir.ranksys.mf.als.PZTFactorizer;
import es.uam.eps.ir.ranksys.mf.plsa.PLSAFactorizer;
import es.uam.eps.ir.ranksys.mf.rec.MFRecommender;
import es.uam.eps.ir.ranksys.nn.item.ItemNeighborhoodRecommender;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.CachedItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.ItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.TopKItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity;
import es.uam.eps.ir.ranksys.nn.item.sim.SetCosineItemSimilarity;
import es.uam.eps.ir.ranksys.nn.item.sim.SetJaccardItemSimilarity;
import es.uam.eps.ir.ranksys.nn.item.sim.VectorCosineItemSimilarity;
import es.uam.eps.ir.ranksys.nn.item.sim.VectorJaccardItemSimilarity;
import es.uam.eps.ir.ranksys.nn.user.UserNeighborhoodRecommender;
import es.uam.eps.ir.ranksys.nn.user.neighborhood.TopKUserNeighborhood;
import es.uam.eps.ir.ranksys.nn.user.neighborhood.UserNeighborhood;
import es.uam.eps.ir.ranksys.nn.user.sim.SetCosineUserSimilarity;
import es.uam.eps.ir.ranksys.nn.user.sim.SetJaccardUserSimilarity;
import es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity;
import es.uam.eps.ir.ranksys.nn.user.sim.VectorCosineUserSimilarity;
import es.uam.eps.ir.ranksys.nn.user.sim.VectorJaccardUserSimilarity;
import es.uam.eps.ir.ranksys.rec.Recommender;
import es.uam.eps.ir.ranksys.rec.fast.basic.PopularityRecommender;
import es.uam.eps.ir.ranksys.rec.fast.basic.RandomRecommender;
import static java.lang.Boolean.parseBoolean;
import static java.lang.Double.parseDouble;
import static java.lang.Integer.parseInt;
import java.util.Iterator;
import java.util.Properties;
import net.recommenders.rival.core.DataModel;
import net.recommenders.rival.recommend.frameworks.AbstractRunner;
import net.recommenders.rival.recommend.frameworks.RecommendationRunner;
import net.recommenders.rival.recommend.frameworks.RecommenderIO;

/**
 * A runner for RankSys' recommenders.
 *
 * @author Saúl Vargas (Saul.Vargas@mendeley.com)
 */
public class RankSysRecommenderRunner extends AbstractRunner<Long, Long> {

    public RankSysRecommenderRunner(Properties props) {
        super(props);
    }

    private String p(String prop) {
        return getProperties().getProperty(prop);
    }

    private String p(String prop, String def) {
        return getProperties().getProperty(prop, def);
    }

    @Override
    public DataModel<Long, Long> run(RUN_OPTIONS opts) throws Exception {
        if (isAlreadyRecommended()) {
            return null;
        }

        FastUserIndex<Long> users = SimpleFastUserIndex.load(p(RecommendationRunner.USER_SET), lp);
        FastItemIndex<Long> items = SimpleFastItemIndex.load(p(RecommendationRunner.ITEM_SET), lp);
        FastPreferenceData<Long, Long> trainingData = SimpleFastPreferenceData.load(p(RecommendationRunner.TRAINING_SET), lp, lp, ddp, users, items);
        FastPreferenceData<Long, Long> testData = SimpleFastPreferenceData.load(p(RecommendationRunner.TEST_SET), lp, lp, ddp, users, items);

        return runRankSysRecommender(opts, trainingData, testData);
    }

    @Override
    public DataModel<Long, Long> run(RUN_OPTIONS opts, DataModel<Long, Long> trainingModel, DataModel<Long, Long> testModel) throws Exception {

        FastUserIndex<Long> users = new SimpleFastUserIndex<Long>() {
            {
                trainingModel.getUsers().forEach(this::add);
                testModel.getUsers().forEach(this::add);
            }
        };
        FastItemIndex<Long> items = new SimpleFastItemIndex<Long>() {
            {
                trainingModel.getItems().forEach(this::add);
                testModel.getItems().forEach(this::add);
            }
        };
        FastPreferenceData<Long, Long> trainingData = new PreferenceDataWrapper<>(trainingModel, users, items);
        FastPreferenceData<Long, Long> testData = new PreferenceDataWrapper<>(testModel, users, items);

        return runRankSysRecommender(opts, trainingData, testData);
    }

    private DataModel<Long, Long> runRankSysRecommender(RUN_OPTIONS opts, FastPreferenceData<Long, Long> trainingData, FastPreferenceData<Long, Long> testData) {

        Recommender<Long, Long> recommender = null;
        int k;
        int q;
        double alpha;
        boolean dense;
        double lambda;
        int numIter;
        switch (p(RecommendationRunner.RECOMMENDER)) {
            case "ub-knn":
                k = parseInt(p(RecommendationRunner.NEIGHBORHOOD, "100"));
                q = parseInt(p("ranksys.ub-knn.q", "1"));
                alpha = parseDouble(p("ranksys.ub-knn.alpha", "0.5"));
                dense = parseBoolean(p("ranksys.ub-knn.dense", "true"));

                UserSimilarity<Long> userSimilarity = null;
                switch (p(RecommendationRunner.SIMILARITY, "vector-cosine")) {
                    case "set-cosine":
                        userSimilarity = new SetCosineUserSimilarity<>(trainingData, alpha, dense);
                        break;
                    case "set-jaccard":
                        userSimilarity = new SetJaccardUserSimilarity<>(trainingData, dense);
                        break;
                    case "cosine":
                    case "vector-cosine":
                        userSimilarity = new VectorCosineUserSimilarity<>(trainingData, alpha, dense);
                        break;
                    case "jaccard":
                    case "vector-jaccard":
                        userSimilarity = new VectorJaccardUserSimilarity<>(trainingData, dense);
                        break;
                }

                UserNeighborhood<Long> userNeighborhood = new TopKUserNeighborhood<>(userSimilarity, k);

                recommender = new UserNeighborhoodRecommender<>(trainingData, userNeighborhood, q);
                break;
            case "ib-nn":
                k = parseInt(p(RecommendationRunner.NEIGHBORHOOD, "10"));
                q = parseInt(p("ranksys.ib-knn.q", "1"));
                alpha = parseDouble(p("ranksys.ib-knn.alpha", "0.5"));
                dense = parseBoolean(p("ranksys.ib-knn.dense", "true"));

                ItemSimilarity<Long> itemSimilarity = null;
                switch (p(RecommendationRunner.SIMILARITY, "vector-cosine")) {
                    case "set-cosine":
                        itemSimilarity = new SetCosineItemSimilarity<>(trainingData, alpha, dense);
                        break;
                    case "set-jaccard":
                        itemSimilarity = new SetJaccardItemSimilarity<>(trainingData, dense);
                        break;
                    case "cosine":
                    case "vector-cosine":
                        itemSimilarity = new VectorCosineItemSimilarity<>(trainingData, alpha, dense);
                        break;
                    case "jaccard":
                    case "vector-jaccard":
                        itemSimilarity = new VectorJaccardItemSimilarity<>(trainingData, dense);
                        break;
                }

                ItemNeighborhood<Long> itemNeighborhood = new TopKItemNeighborhood<>(itemSimilarity, k);
                itemNeighborhood = new CachedItemNeighborhood<>(itemNeighborhood);

                recommender = new ItemNeighborhoodRecommender<>(trainingData, itemNeighborhood, q);
                break;
            case "mf":
                k = parseInt(p(RecommendationRunner.FACTORS, "50"));

                Factorizer<Long, Long> factorizer = null;
                switch (p(RecommendationRunner.FACTORIZER, "hkv")) {
                    case "hkv":
                        lambda = parseDouble(p("ranksys.mf.lambda", "0.1"));
                        alpha = parseDouble(p("ranksys.mf.alpha", "1.0"));
                        numIter = parseInt(p(RecommendationRunner.ITERATIONS, "20"));
                        factorizer = new HKVFactorizer<>(lambda, x -> 1 + alpha * x, numIter);
                        break;
                    case "pzt":
                        lambda = parseDouble(p("ranksys.mf.lambda", "0.1"));
                        alpha = parseDouble(p("ranksys.mf.alpha", "1.0"));
                        numIter = parseInt(p(RecommendationRunner.ITERATIONS, "20"));
                        factorizer = new PZTFactorizer<>(lambda, x -> 1 + alpha * x, numIter);
                        break;
                    case "plsa":
                        numIter = parseInt(p(RecommendationRunner.ITERATIONS, "100"));
                        factorizer = new PLSAFactorizer<>(numIter);
                        break;
                }

                Factorization<Long, Long> factorization = factorizer.factorize(k, trainingData);

                recommender = new MFRecommender<>(trainingData, trainingData, factorization);
                break;
            case "pop":
                recommender = new PopularityRecommender<>(trainingData);
                break;
            case "rnd":
                recommender = new RandomRecommender<>(trainingData, trainingData);
                break;
        }

        DataModel<Long, Long> model;
        switch (opts) {
            case RETURN_AND_OUTPUT_RECS:
            case RETURN_RECS:
                model = new DataModel<>();
                break;
            default:
                model = null;
        }
        String name;
        switch (opts) {
            case RETURN_AND_OUTPUT_RECS:
            case OUTPUT_RECS:
                name = getFileName();
                break;
            default:
                name = null;
        }
        boolean createFile = true;
        Iterator<Long> it = testData.getUsersWithPreferences().iterator();
        while (it.hasNext()) {
            Long user = it.next();
            Recommendation<Long, Long> recommendation = recommender.getRecommendation(user, 0);
            RecommenderIO.writeData(user, recommendation.getItems(), getPath(), name, !createFile, model);
            createFile = false;
        }
        return model;
    }

}
